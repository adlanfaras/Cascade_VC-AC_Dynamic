from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from .control import get_path, set_path
from .control import ControlSystem
from .model import CascadeSystemModel
from .numerics import NewtonSolveError, newton_raphson_fd


STARTUP_CACHE_VERSION = 3

STATE_INDEX = {
    "room_c": 0,
    "sink_c": 1,
    "t3_c": 2,
    "t4_c": 3,
    "t6_c": 4,
    "tevap_c": 5,
    "tcond_c": 6,
    "m_ref_kg_s": 7,
}

DEFAULT_STARTUP_FREE_PARAMETERS = [
    {"path": "vcc_cycle.condenser_ua_w_k", "min": 1000.0, "max": 100000.0},
    {"path": "vcc_cycle.cascade_ua_w_k", "min": 100.0, "max": 100000.0},
    {"path": "boundary_conditions.load_before_w", "min": 1000.0, "max": 200000.0},
    {"path": "vcc_cycle.expansion_valve.opening", "min": 0.05, "max": 1.0, "freeze_in_transient": False},
    {"path": "air_cycle.pressure_ratio", "min": 1.01, "max": 1.6},
    {"path": "air_cycle.compressor_mass_flow.speed_rpm", "min": 10000.0, "max": 20000.0, "freeze_in_transient": False},
]
DEFAULT_STARTUP_FREE_STATE_VARIABLES = ["t3_c", "t4_c", "t6_c", "m_ref_kg_s"]


@dataclass
class StartupInitializationResult:
    unknowns: np.ndarray
    iterations: int
    cost: float
    snapshot: dict[str, float]
    cache_hit: bool = False
    cache_path: str | None = None


def _log(message: str) -> None:
    print(message, flush=True)


def _startup_signature_data(config: dict[str, Any]) -> dict[str, Any]:
    sim_cfg = config["simulation"].get("startup_initialization", {})
    return {
        "cache_version": STARTUP_CACHE_VERSION,
        "startup_initialization": sim_cfg,
        "fluids": config["fluids"],
        "air_cycle": config["air_cycle"],
        "vcc_cycle": config["vcc_cycle"],
        "boundary_conditions": config["boundary_conditions"],
        "thermal_masses": config["thermal_masses"],
        "initial_guess": config["initial_guess"],
    }


def _startup_cache_signature(config: dict[str, Any]) -> str:
    payload = json.dumps(_startup_signature_data(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _startup_cache_path(config: dict[str, Any]) -> Path:
    plot_path = Path(config["output"]["plot_file"])
    return plot_path.parent / "startup_cache" / f"{_startup_cache_signature(config)}.json"


def initial_vector(config: dict) -> np.ndarray:
    guess = config["initial_guess"]
    return np.array(
        [
            guess["room_c"],
            guess["sink_c"],
            guess["t3_c"],
            guess["t4_c"],
            guess["t6_c"],
            guess["tevap_c"],
            guess["tcond_c"],
            guess["m_ref_kg_s"],
        ],
        dtype=float,
    )


def _startup_target_state(config: dict[str, Any], targets: dict[str, float]) -> np.ndarray:
    unknowns = initial_vector(config)
    for key, value in targets.items():
        if key in STATE_INDEX:
            unknowns[STATE_INDEX[key]] = float(value)
    return unknowns


def _startup_scales(config: dict[str, Any], targets: dict[str, float]) -> dict[str, float]:
    room_load = abs(config["boundary_conditions"]["load_before_w"])
    dock_load = abs(config["boundary_conditions"]["dock_load_before_w"])
    return {
        "w": max(room_load + dock_load, 1.0),
        "kg_s": max(abs(targets.get("m_ref_kg_s", config["initial_guess"]["m_ref_kg_s"])), 1.0e-3),
        "temperature_c": 5.0,
        "delta_t_c": 2.0,
    }


def _startup_free_parameters(startup_cfg: dict[str, Any]) -> list[dict[str, float]]:
    return startup_cfg.get("free_parameters", DEFAULT_STARTUP_FREE_PARAMETERS)


def _free_parameter_initial_value(config: dict[str, Any], item: dict[str, Any]) -> float:
    return float(item.get("initial", get_path(config, item["path"])))


def _load_startup_cache(cache_path: Path, signature: str) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("cache_version") != STARTUP_CACHE_VERSION:
        return None
    if payload.get("signature") != signature:
        return None
    return payload


def _save_startup_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _apply_cached_free_parameters(config: dict[str, Any], free_parameter_values: dict[str, float]) -> None:
    for path, value in free_parameter_values.items():
        set_path(config, path, value)


def configure_disturbances(config: dict[str, Any]) -> None:
    infiltration_cfg = config.get("disturbances", {}).get("infiltration", {})
    if not infiltration_cfg.get("enabled", False):
        return

    mode = infiltration_cfg.get("magnitude_mode", "fixed_w")
    if mode == "fixed_w":
        infiltration_cfg["resolved_magnitude_w"] = float(infiltration_cfg.get("magnitude_w", 0.0))
        return
    if mode != "percent_of_room_load":
        raise ValueError(f"Unsupported infiltration magnitude_mode: {mode}")

    percentage = float(infiltration_cfg.get("load_percentage", 0.10))
    reference_path = infiltration_cfg.get("reference_load_path", "boundary_conditions.load_before_w")
    reference_load_w = float(get_path(config, reference_path))
    infiltration_cfg["resolved_magnitude_w"] = percentage * reference_load_w


def solve_startup_initialization(config: dict, model: CascadeSystemModel) -> StartupInitializationResult:
    sim_cfg = config["simulation"]
    startup_cfg = sim_cfg.get("startup_initialization", {})
    time_s = startup_cfg.get("time_s", sim_cfg["t_start_s"])
    mode = startup_cfg.get("mode", "design_point")
    signature = _startup_cache_signature(config)
    cache_path = _startup_cache_path(config)
    use_cache = startup_cfg.get("use_cache", True)

    if use_cache:
        cached = _load_startup_cache(cache_path, signature)
        if cached is not None:
            _log(f"[startup] cache hit: {cache_path}")
            free_parameter_values = {str(path): float(value) for path, value in cached.get("free_parameter_values", {}).items()}
            _apply_cached_free_parameters(config, free_parameter_values)
            unknowns = np.asarray(cached["unknowns"], dtype=float)
            snapshot = {str(key): float(value) for key, value in cached.get("snapshot", {}).items()}
            _log(
                f"[startup] loaded cached solution | iterations={int(cached.get('iterations', 0))} | "
                f"cost={float(cached.get('cost', 0.0)):.3e}"
            )
            return StartupInitializationResult(
                unknowns=unknowns,
                iterations=int(cached.get("iterations", 0)),
                cost=float(cached.get("cost", 0.0)),
                snapshot=snapshot,
                cache_hit=True,
                cache_path=str(cache_path),
            )

    _log(f"[startup] solving mode={mode}...")

    if mode == "state_only":
        def residual_fn(x: np.ndarray) -> np.ndarray:
            return model.steady_state_residual(x, time_s)

        unknowns, iterations = newton_raphson_fd(
            residual_fn,
            initial_vector(config),
            tol=startup_cfg.get("newton_tol", sim_cfg["newton_tol"]),
            max_iter=startup_cfg.get("newton_max_iter", sim_cfg["newton_max_iter"]),
            step=startup_cfg.get("fd_step", sim_cfg["fd_step"]),
        )
        snapshot = model.startup_metrics(unknowns, time_s)
        result = StartupInitializationResult(unknowns, iterations, 0.0, snapshot, cache_hit=False, cache_path=str(cache_path))
        if use_cache:
            _save_startup_cache(
                cache_path,
                {
                    "cache_version": STARTUP_CACHE_VERSION,
                    "signature": signature,
                    "iterations": result.iterations,
                    "cost": result.cost,
                    "unknowns": result.unknowns.tolist(),
                    "snapshot": result.snapshot,
                    "free_parameter_values": {},
                },
            )
            _log(f"[startup] cache saved: {cache_path}")
        _log(
            f"[startup] solved state_only | iterations={iterations} | cost=0.000e+00 | "
            f"room={snapshot['room_c']:.2f} C | m_ref={snapshot['m_ref_kg_s']:.4f} kg/s"
        )
        return result

    targets = startup_cfg.get("targets", {})
    fixed_unknowns = _startup_target_state(config, targets)
    free_state_keys = startup_cfg.get("free_state_variables", DEFAULT_STARTUP_FREE_STATE_VARIABLES)
    free_parameters = _startup_free_parameters(startup_cfg)
    scales = _startup_scales(config, targets)

    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    x_scale: list[float] = []

    for key in free_state_keys:
        x0.append(float(fixed_unknowns[STATE_INDEX[key]]))
        lower.append(float(startup_cfg.get("temperature_min_c", -80.0)))
        upper.append(float(startup_cfg.get("temperature_max_c", 90.0)))
        x_scale.append(max(abs(x0[-1]), 10.0))

    for item in free_parameters:
        value = _free_parameter_initial_value(config, item)
        x0.append(value)
        lower.append(float(item.get("min", -np.inf)))
        upper.append(float(item.get("max", np.inf)))
        x_scale.append(float(item.get("scale", max(abs(value), 1.0))))

    def apply_design_variables(x: np.ndarray) -> np.ndarray:
        unknowns = fixed_unknowns.copy()
        offset = 0
        mirror_room_load = config["boundary_conditions"].get("load_after_w") == config["boundary_conditions"].get("load_before_w")
        for key in free_state_keys:
            unknowns[STATE_INDEX[key]] = x[offset]
            offset += 1
        for item in free_parameters:
            set_path(config, item["path"], x[offset])
            offset += 1
        if mirror_room_load and any(item["path"] == "boundary_conditions.load_before_w" for item in free_parameters):
            config["boundary_conditions"]["load_after_w"] = config["boundary_conditions"]["load_before_w"]
        return unknowns

    def scaled_balance_residual_fn(x: np.ndarray) -> np.ndarray:
        unknowns = apply_design_variables(x)
        balance = model.startup_balance_residual(unknowns, time_s)
        return np.array(
            [
                balance[0] / scales["w"],
                balance[1] / scales["w"],
                balance[2] / scales["w"],
                balance[3] / scales["w"],
                balance[4] / scales["w"],
                balance[5] / scales["w"],
                balance[6] / scales["kg_s"],
                balance[7] / scales["kg_s"],
            ],
            dtype=float,
        )

    room_delta_t_target_c = startup_cfg.get("room_delta_t_target_c")
    t5_target_c = startup_cfg.get("t5_target_c")
    startup_residual_size = 8
    if room_delta_t_target_c is not None:
        startup_residual_size += 1
    if t5_target_c is not None:
        startup_residual_size += 1

    def startup_residual_fn(x: np.ndarray) -> np.ndarray:
        try:
            residual = scaled_balance_residual_fn(x)
        except ValueError:
            return np.full(startup_residual_size, 1.0e6, dtype=float)
        if room_delta_t_target_c is None:
            return residual
        unknowns = apply_design_variables(x)
        try:
            metrics = model.startup_metrics(unknowns, time_s)
        except ValueError:
            return np.full(startup_residual_size, 1.0e6, dtype=float)
        extra_residuals: list[float] = []
        if t5_target_c is not None:
            extra_residuals.append((metrics["t5_c"] - float(t5_target_c)) / scales["temperature_c"])
        if room_delta_t_target_c is not None:
            extra_residuals.append(((metrics["room_c"] - metrics["t5_c"]) - float(room_delta_t_target_c)) / scales["delta_t_c"])
        return np.concatenate([residual, np.asarray(extra_residuals, dtype=float)])

    result = least_squares(
        startup_residual_fn,
        np.array(x0, dtype=float),
        bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
        x_scale=np.array(x_scale, dtype=float),
        ftol=startup_cfg.get("least_squares_tol", 1.0e-10),
        xtol=startup_cfg.get("least_squares_tol", 1.0e-10),
        gtol=startup_cfg.get("least_squares_tol", 1.0e-10),
        max_nfev=startup_cfg.get("max_function_evals", 500),
    )
    if not result.success:
        raise RuntimeError(f"Startup initialization did not converge: {result.message}")
    x_solution = np.asarray(result.x, dtype=float)
    startup_iterations = int(result.nfev)
    startup_cost = float(result.cost)

    unknowns = apply_design_variables(x_solution)
    snapshot = model.startup_metrics(unknowns, time_s)
    free_parameter_values = {item["path"]: float(get_path(config, item["path"])) for item in free_parameters}
    for path, value in free_parameter_values.items():
        snapshot[f"startup_solved_{path.replace('.', '_')}"] = value

    result = StartupInitializationResult(
        unknowns=unknowns,
        iterations=int(startup_iterations),
        cost=float(startup_cost),
        snapshot=snapshot,
        cache_hit=False,
        cache_path=str(cache_path),
    )
    if use_cache:
        _save_startup_cache(
            cache_path,
            {
                "cache_version": STARTUP_CACHE_VERSION,
                "signature": signature,
                "iterations": result.iterations,
                "cost": result.cost,
                "unknowns": result.unknowns.tolist(),
                "snapshot": result.snapshot,
                "free_parameter_values": free_parameter_values,
            },
        )
        _log(f"[startup] cache saved: {cache_path}")
    _log(
        f"[startup] solved design_point | iterations={result.iterations} | cost={result.cost:.3e} | "
        f"room={snapshot['room_c']:.2f} C | m_ref={snapshot['m_ref_kg_s']:.4f} kg/s | m_air={snapshot['m_air_kg_s']:.4f} kg/s"
    )
    return result


def solve_dynamic_step(residual_fn, x0: np.ndarray, sim_cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    try:
        return newton_raphson_fd(
            residual_fn,
            x0,
            tol=sim_cfg["newton_tol"],
            max_iter=sim_cfg["newton_max_iter"],
            step=sim_cfg["fd_step"],
        )
    except NewtonSolveError:
        result = least_squares(
            residual_fn,
            np.asarray(x0, dtype=float),
            x_scale=np.maximum(np.abs(np.asarray(x0, dtype=float)), 1.0),
            ftol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            xtol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            gtol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            max_nfev=sim_cfg.get("fallback_max_function_evals", 500),
        )
        if not result.success:
            raise
        return np.asarray(result.x, dtype=float), int(result.nfev)


def run_simulation(config: dict) -> list[dict[str, float]]:
    plant_config = deepcopy(config)
    sim_cfg = plant_config["simulation"]
    model = CascadeSystemModel(plant_config)
    progress_interval = int(sim_cfg.get("progress_interval_steps", 20))

    dt_s = sim_cfg["dt_s"]
    times = np.arange(sim_cfg["t_start_s"], sim_cfg["t_end_s"] + dt_s, dt_s)
    startup_cfg = sim_cfg.get("startup_initialization", {})
    startup_enabled = startup_cfg.get("enabled", True)
    startup_iters = 0
    startup_cost = 0.0
    startup_snapshot: dict[str, float] = {}
    frozen_actuator_paths: set[str] = set()
    if startup_enabled:
        startup = solve_startup_initialization(plant_config, model)
        unknowns = startup.unknowns
        startup_iters = startup.iterations
        startup_cost = startup.cost
        startup_snapshot = startup.snapshot
        if startup_cfg.get("freeze_solved_parameters", True):
            frozen_actuator_paths = {
                item["path"]
                for item in _startup_free_parameters(startup_cfg)
                if item.get("freeze_in_transient", True)
            }
    else:
        unknowns = initial_vector(plant_config)

    if startup_snapshot:
        for controller in plant_config.get("control", {}).get("controllers", []):
            setpoint_from_startup = controller.get("setpoint_from_startup")
            if setpoint_from_startup and setpoint_from_startup in startup_snapshot:
                controller["setpoint"] = float(startup_snapshot[setpoint_from_startup])
            solved_bias_key = f"startup_solved_{controller['actuator_path'].replace('.', '_')}"
            if controller.get("bias_from_startup", True) and solved_bias_key in startup_snapshot:
                controller["bias"] = float(startup_snapshot[solved_bias_key])

    configure_disturbances(plant_config)
    control = ControlSystem(plant_config, frozen_actuator_paths=frozen_actuator_paths)
    state = unknowns[:2].copy()
    history: list[dict[str, float]] = []

    _log(f"[run] steps={len(times) - 1} dt={dt_s:.1f}s interval={progress_interval}")
    for idx, time_s in enumerate(times):
        if idx == 0:
            step = model.post_process(unknowns, time_s)
            step.values["startup_initialization_enabled"] = float(startup_enabled)
            step.values["startup_iterations"] = float(startup_iters)
            step.values["startup_newton_iterations"] = float(startup_iters)
            step.values["startup_cost"] = startup_cost
            for key, value in startup_snapshot.items():
                if key.startswith("startup_solved_"):
                    step.values[key] = value
            step.values["newton_iterations"] = 0.0
            history.append(step.values)
            _log(
                f"[startup] t={time_s / 60.0:.2f} min | room={step.values['room_c']:.2f} C | "
                f"m_ref={step.values['m_ref_kg_s']:.4f} kg/s | m_air={step.values['m_air_kg_s']:.4f} kg/s | "
                f"COP={step.values['cop_system']:.3f}"
            )
            continue

        prev_state = state.copy()
        controller_outputs = control.update(history[-1], plant_config, dt_s)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            return model.residual(x, prev_state, time_s, dt_s)

        unknowns, iters = solve_dynamic_step(residual_fn, unknowns, sim_cfg)
        step = model.post_process(unknowns, time_s)
        step.values["newton_iterations"] = float(iters)
        step.values.update(controller_outputs)
        history.append(step.values)
        state = step.state_vector
        if idx % max(progress_interval, 1) == 0 or idx == len(times) - 1:
            _log(
                f"[run] t={time_s / 60.0:.2f} min | room={step.values['room_c']:.2f} C | "
                f"m_ref={step.values['m_ref_kg_s']:.4f} kg/s | m_air={step.values['m_air_kg_s']:.4f} kg/s | "
                f"COP={step.values['cop_system']:.3f} | newton={iters}"
            )

    return history


def save_plot(history: list[dict[str, float]], plot_file: str | Path) -> None:
    out_path = Path(plot_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_min = np.array([row["time_s"] for row in history]) / 60.0
    room_c = np.array([row["room_c"] for row in history])
    t3_c = np.array([row["t3_c"] for row in history])
    m_ref_kg_s = np.array([row["m_ref_kg_s"] for row in history])
    m_air_kg_s = np.array([row["m_air_kg_s"] for row in history])
    cop = np.array([row["cop_system"] for row in history])

    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(t_min, room_c, label="Room")
    axes[0].set_ylabel("Temperature [C]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, t3_c, label="Air after cascade exchanger")
    axes[1].set_ylabel("Temperature [C]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_min, m_ref_kg_s, label="Refrigerant mass flow")
    axes[2].set_ylabel("Mass Flow [kg/s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_min, m_air_kg_s, label="Air mass flow")
    axes[3].set_ylabel("Mass Flow [kg/s]")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t_min, cop, label="COP")
    axes[4].set_xlabel("Time [min]")
    axes[4].set_ylabel("COP")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_csv(history: list[dict[str, float]], csv_file: str | Path) -> None:
    out_path = Path(csv_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
