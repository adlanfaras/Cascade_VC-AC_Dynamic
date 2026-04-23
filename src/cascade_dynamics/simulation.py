from __future__ import annotations

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
from .numerics import newton_raphson_fd


STATE_INDEX = {
    "room_c": 0,
    "sink_c": 1,
    "t3_c": 2,
    "t4_c": 3,
    "t6_c": 4,
    "tevap_c": 5,
    "m_ref_kg_s": 6,
}

DEFAULT_STARTUP_FREE_PARAMETERS = [
    {"path": "vcc_cycle.condenser_ua_w_k", "min": 1000.0, "max": 100000.0},
    {"path": "vcc_cycle.cascade_ua_w_k", "min": 100.0, "max": 100000.0},
    {"path": "vcc_cycle.expansion_valve.opening", "min": 0.05, "max": 1.0},
    {"path": "vcc_cycle.compressor_pressure_ratio", "min": 3.0, "max": 6.0},
    {"path": "air_cycle.pressure_ratio", "min": 1.01, "max": 1.6},
    {"path": "air_cycle.compressor_mass_flow.speed_fraction", "min": 0.5, "max": 1.5},
]


@dataclass
class StartupInitializationResult:
    unknowns: np.ndarray
    iterations: int
    cost: float
    snapshot: dict[str, float]


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
    }


def _startup_free_parameters(startup_cfg: dict[str, Any]) -> list[dict[str, float]]:
    return startup_cfg.get("free_parameters", DEFAULT_STARTUP_FREE_PARAMETERS)


def _free_parameter_initial_value(config: dict[str, Any], item: dict[str, Any]) -> float:
    return float(item.get("initial", get_path(config, item["path"])))


def solve_startup_initialization(config: dict, model: CascadeSystemModel) -> StartupInitializationResult:
    sim_cfg = config["simulation"]
    startup_cfg = sim_cfg.get("startup_initialization", {})
    time_s = startup_cfg.get("time_s", sim_cfg["t_start_s"])
    mode = startup_cfg.get("mode", "design_point")

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
        step = model.post_process(unknowns, time_s)
        return StartupInitializationResult(unknowns, iterations, 0.0, step.values)

    targets = startup_cfg.get("targets", {})
    fixed_unknowns = _startup_target_state(config, targets)
    free_state_keys = startup_cfg.get("free_state_variables", ["t3_c", "t4_c", "t6_c"])
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
        for key in free_state_keys:
            unknowns[STATE_INDEX[key]] = x[offset]
            offset += 1
        for item in free_parameters:
            set_path(config, item["path"], x[offset])
            offset += 1
        return unknowns

    def residual_fn(x: np.ndarray) -> np.ndarray:
        unknowns = apply_design_variables(x)
        balance = model.startup_balance_residual(unknowns, time_s)
        residuals = [
            balance[0] / scales["w"],
            balance[1] / scales["w"],
            balance[2] / scales["w"],
            balance[3] / scales["w"],
            balance[4] / scales["w"],
            balance[5] / scales["w"],
            balance[6] / scales["kg_s"],
        ]
        step = model.post_process(unknowns, time_s)
        for key in ("tcond_c", "m_air_kg_s"):
            if key in targets:
                scale = scales["temperature_c"] if key.endswith("_c") else max(abs(targets[key]), 1.0e-3)
                residuals.append((step.values[key] - float(targets[key])) / scale)
        return np.array(residuals, dtype=float)

    result = least_squares(
        residual_fn,
        np.array(x0, dtype=float),
        bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
        x_scale=np.array(x_scale, dtype=float),
        ftol=startup_cfg.get("least_squares_tol", 1.0e-10),
        xtol=startup_cfg.get("least_squares_tol", 1.0e-10),
        gtol=startup_cfg.get("least_squares_tol", 1.0e-10),
        max_nfev=startup_cfg.get("max_function_evals", 500),
    )
    unknowns = apply_design_variables(result.x)
    step = model.post_process(unknowns, time_s)
    snapshot = step.values.copy()
    for item in free_parameters:
        snapshot[f"startup_solved_{item['path'].replace('.', '_')}"] = float(get_path(config, item["path"]))
    if not result.success:
        raise RuntimeError(f"Startup initialization did not converge: {result.message}")

    return StartupInitializationResult(
        unknowns=unknowns,
        iterations=int(result.nfev),
        cost=float(result.cost),
        snapshot=snapshot,
    )


def run_simulation(config: dict) -> list[dict[str, float]]:
    sim_cfg = config["simulation"]
    model = CascadeSystemModel(config)

    dt_s = sim_cfg["dt_s"]
    times = np.arange(sim_cfg["t_start_s"], sim_cfg["t_end_s"] + dt_s, dt_s)
    startup_cfg = sim_cfg.get("startup_initialization", {})
    startup_enabled = startup_cfg.get("enabled", True)
    startup_iters = 0
    startup_cost = 0.0
    startup_snapshot: dict[str, float] = {}
    frozen_actuator_paths: set[str] = set()
    if startup_enabled:
        startup = solve_startup_initialization(config, model)
        unknowns = startup.unknowns
        startup_iters = startup.iterations
        startup_cost = startup.cost
        startup_snapshot = startup.snapshot
        if startup_cfg.get("freeze_solved_parameters", True):
            frozen_actuator_paths = {item["path"] for item in _startup_free_parameters(startup_cfg)}
    else:
        unknowns = initial_vector(config)
    control = ControlSystem(config, frozen_actuator_paths=frozen_actuator_paths)
    state = unknowns[:2].copy()
    history: list[dict[str, float]] = []

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
            continue

        prev_state = state.copy()
        controller_outputs = control.update(history[-1], config, dt_s)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            return model.residual(x, prev_state, time_s, dt_s)

        unknowns, iters = newton_raphson_fd(
            residual_fn,
            unknowns,
            tol=sim_cfg["newton_tol"],
            max_iter=sim_cfg["newton_max_iter"],
            step=sim_cfg["fd_step"],
        )
        step = model.post_process(unknowns, time_s)
        step.values["newton_iterations"] = float(iters)
        step.values.update(controller_outputs)
        history.append(step.values)
        state = step.state_vector

    return history


def save_plot(history: list[dict[str, float]], plot_file: str | Path) -> None:
    out_path = Path(plot_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_min = np.array([row["time_s"] for row in history]) / 60.0
    room_c = np.array([row["room_c"] for row in history])
    m_ref_kg_s = np.array([row["m_ref_kg_s"] for row in history])
    m_air_kg_s = np.array([row["m_air_kg_s"] for row in history])
    cop = np.array([row["cop_system"] for row in history])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t_min, room_c, label="Room")
    axes[0].set_ylabel("Temperature [C]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, m_ref_kg_s, label="Refrigerant mass flow")
    axes[1].set_ylabel("Mass Flow [kg/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_min, m_air_kg_s, label="Air mass flow")
    axes[2].set_ylabel("Mass Flow [kg/s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_min, cop, label="COP")
    axes[3].set_xlabel("Time [min]")
    axes[3].set_ylabel("COP")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

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
