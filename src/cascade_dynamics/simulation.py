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
from CoolProp.CoolProp import PropsSI
from scipy.optimize import least_squares

from .compressor_map import ammonia_compressor_map
from .control import get_path, set_path
from .control import ControlSystem
from .components import positive_lmtd
from .fluids import h_refrigerant_liquid, p_sat
from .model import CascadeSystemModel
from .numerics import NewtonSolveError, newton_raphson_fd


KELVIN_OFFSET = 273.15
STARTUP_CACHE_VERSION = 8

STATE_INDEX = {
    "room_c": 0,
    "sink_c": 1,
    "t3_c": 2,
    "t4_c": 3,
    "t6_c": 4,
    "tevap_c": 5,
    "tcond_c": 6,
    "m_ref_kg_s": 7,
    "dock_c": 8,
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
PAPER_DESIGN_SOLVED_PATHS = [
    "air_cycle.pressure_ratio",
    "air_cycle.compressor_mass_flow.speed_rpm",
    "air_cycle.regenerator_ua_w_k",
    "vcc_cycle.cascade_ua_w_k",
    "vcc_cycle.condenser_ua_w_k",
    "vcc_cycle.compressor.speed_rpm",
    "vcc_cycle.expansion_valve.opening",
    "boundary_conditions.load_before_w",
    "boundary_conditions.load_after_w",
    "boundary_conditions.sink_m_dot_kg_s",
]


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
            guess.get("dock_c", config["boundary_conditions"]["dock_initial_c"]),
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
    elif mode == "percent_of_room_load":
        percentage = float(infiltration_cfg.get("load_percentage", 0.10))
        reference_path = infiltration_cfg.get("reference_load_path", "boundary_conditions.load_before_w")
        reference_load_w = float(get_path(config, reference_path))
        infiltration_cfg["resolved_magnitude_w"] = percentage * reference_load_w
    else:
        raise ValueError(f"Unsupported infiltration magnitude_mode: {mode}")


def _free_parameter_bounds(startup_cfg: dict[str, Any], path: str, default: tuple[float, float]) -> tuple[float, float]:
    for item in _startup_free_parameters(startup_cfg):
        if item["path"] == path:
            return float(item.get("min", default[0])), float(item.get("max", default[1]))
    return default


def _set_and_mirror_room_load(config: dict[str, Any], value_w: float) -> None:
    mirror_room_load = config["boundary_conditions"].get("load_after_w") == config["boundary_conditions"].get("load_before_w")
    config["boundary_conditions"]["load_before_w"] = float(value_w)
    if mirror_room_load:
        config["boundary_conditions"]["load_after_w"] = float(value_w)


def _solve_scalar_bisection(
    residual_fn,
    lower: float,
    upper: float,
    *,
    tol: float = 1.0e-7,
    max_iter: int = 100,
) -> tuple[float, int]:
    lo = float(lower)
    hi = float(upper)
    f_lo = float(residual_fn(lo))
    f_hi = float(residual_fn(hi))
    if abs(f_lo) <= tol:
        return lo, 0
    if abs(f_hi) <= tol:
        return hi, 0
    if f_lo * f_hi > 0.0:
        raise ValueError(f"Could not bracket scalar startup solve: f({lo})={f_lo}, f({hi})={f_hi}")
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = float(residual_fn(mid))
        if abs(f_mid) <= tol or abs(hi - lo) <= tol:
            return mid, iteration
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi), max_iter


def _solve_air_pressure_ratio_for_t5(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    t5_target_c: float,
) -> int:
    room_c, _, t3_c, t4_c, t6_c, _, _, _, _ = unknowns
    lower, upper = _free_parameter_bounds(startup_cfg, "air_cycle.pressure_ratio", (1.01, 2.0))

    def residual(pressure_ratio: float) -> float:
        config["air_cycle"]["pressure_ratio"] = pressure_ratio
        air = model._evaluate_air_cycle(
            room_c + KELVIN_OFFSET,
            t3_c + KELVIN_OFFSET,
            t4_c + KELVIN_OFFSET,
            t6_c + KELVIN_OFFSET,
        )
        return air["t5_k"] - KELVIN_OFFSET - t5_target_c

    pressure_ratio, iterations = _solve_scalar_bisection(residual, lower, upper, tol=1.0e-6)
    config["air_cycle"]["pressure_ratio"] = float(pressure_ratio)
    return iterations


def _solve_air_speed_for_mass_flow(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    target_m_air_kg_s: float,
) -> int:
    room_c, _, t3_c, t4_c, t6_c, _, _, _, _ = unknowns
    speed_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "air_cycle.compressor_mass_flow.speed_rpm",
        (float(speed_cfg.get("speed_rpm", 15000.0)), float(speed_cfg.get("speed_rpm", 15000.0))),
    )

    def residual(speed_rpm: float) -> float:
        speed_cfg["speed_rpm"] = speed_rpm
        air = model._evaluate_air_cycle(
            room_c + KELVIN_OFFSET,
            t3_c + KELVIN_OFFSET,
            t4_c + KELVIN_OFFSET,
            t6_c + KELVIN_OFFSET,
        )
        return air["m_air"] - target_m_air_kg_s

    speed_rpm, iterations = _solve_scalar_bisection(residual, lower, upper, tol=1.0e-7)
    speed_cfg["speed_rpm"] = float(speed_rpm)
    return iterations


def _solve_air_speed_for_evaporator_capacity(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    target_capacity_w: float,
) -> int:
    room_c, _, t3_c, t4_c, t6_c, tevap_c, _, _, dock_c = unknowns
    speed_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "air_cycle.compressor_mass_flow.speed_rpm",
        (float(speed_cfg.get("speed_rpm", 15000.0)), float(speed_cfg.get("speed_rpm", 15000.0))),
    )
    x0 = np.array([min(max(float(speed_cfg.get("speed_rpm", 15000.0)), lower), upper)], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        speed_cfg["speed_rpm"] = float(x[0])
        air = model._evaluate_air_cycle(
            room_c + KELVIN_OFFSET,
            t3_c + KELVIN_OFFSET,
            t4_c + KELVIN_OFFSET,
            t6_c + KELVIN_OFFSET,
        )
        q_dock = model._evaluate_dock_evaporator(dock_c, tevap_c)
        return np.array([(air["q_cascade"] + q_dock - target_capacity_w) / max(target_capacity_w, 1.0)], dtype=float)

    result = least_squares(
        residual,
        x0,
        bounds=(np.array([lower], dtype=float), np.array([upper], dtype=float)),
        x_scale=np.array([max(abs(x0[0]), 1.0)], dtype=float),
        ftol=startup_cfg.get("least_squares_tol", 1.0e-10),
        xtol=startup_cfg.get("least_squares_tol", 1.0e-10),
        gtol=startup_cfg.get("least_squares_tol", 1.0e-10),
        max_nfev=startup_cfg.get("max_function_evals", 500),
    )
    if not result.success:
        raise RuntimeError(f"Could not solve air speed for target evaporator capacity: {result.message}")
    speed_cfg["speed_rpm"] = float(result.x[0])
    final_residual_w = float(residual(result.x)[0] * max(target_capacity_w, 1.0))
    if abs(final_residual_w) > float(startup_cfg.get("capacity_target_tolerance_w", 1.0)):
        raise RuntimeError(
            f"Could not meet target evaporator capacity {target_capacity_w:.3f} W; "
            f"best residual is {final_residual_w:.3f} W at {float(result.x[0]):.3f} rpm."
        )
    return int(result.nfev)


def _solve_refrigerant_speed_for_mass_flow(
    config: dict[str, Any],
    startup_cfg: dict[str, Any],
    tevap_k: float,
    tcond_k: float,
    target_m_ref_kg_s: float,
) -> int:
    compressor_cfg = config["vcc_cycle"]["compressor"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "vcc_cycle.compressor.speed_rpm",
        (float(compressor_cfg.get("speed_min_rpm", compressor_cfg["speed_rpm"])), float(compressor_cfg.get("speed_max_rpm", compressor_cfg["speed_rpm"]))),
    )
    check_range = compressor_cfg.get("check_range", True)

    def residual(speed_rpm: float) -> float:
        return ammonia_compressor_map(tcond_k, tevap_k, speed_rpm, check_range=check_range)["mdot_kg_s"] - target_m_ref_kg_s

    speed_rpm, iterations = _solve_scalar_bisection(residual, lower, upper, tol=1.0e-8)
    compressor_cfg["speed_rpm"] = float(speed_rpm)
    return iterations


def _build_paper_design_unknowns(config: dict[str, Any], startup_cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
    targets = startup_cfg.get("targets", {})
    room_c = float(targets.get("room_c", config["initial_guess"].get("room_c", -30.0)))
    dock_c = float(targets.get("dock_c", config["boundary_conditions"].get("dock_initial_c", 5.0)))
    sink_c = float(targets.get("sink_c", config["initial_guess"].get("sink_c", 35.0)))
    tevap_c = float(targets.get("tevap_c", config["initial_guess"].get("tevap_c", -5.0)))
    tcond_c = float(targets.get("tcond_c", config["initial_guess"].get("tcond_c", 40.0)))
    room_delta_t_c = float(startup_cfg.get("room_delta_t_target_c", 10.0))
    cascade_delta_t_c = float(startup_cfg.get("cascade_air_evap_min_delta_t_c", 10.0))
    regenerator_effectiveness = float(startup_cfg.get("regenerator_effectiveness", 0.9))

    t3_c = tevap_c + cascade_delta_t_c
    t4_c = t3_c - regenerator_effectiveness * (t3_c - room_c)
    t6_c = room_c + regenerator_effectiveness * (t3_c - room_c)
    t5_target_c = float(startup_cfg.get("t5_target_c", room_c - room_delta_t_c))
    m_ref_kg_s = float(config["initial_guess"].get("m_ref_kg_s", 0.1))

    return (
        np.array([room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref_kg_s, dock_c], dtype=float),
        t5_target_c,
    )


def _solve_paper_design_initialization(config: dict[str, Any], model: CascadeSystemModel, startup_cfg: dict[str, Any], time_s: float) -> StartupInitializationResult:
    unknowns, t5_target_c = _build_paper_design_unknowns(config, startup_cfg)
    iteration_count = _solve_air_pressure_ratio_for_t5(config, model, startup_cfg, unknowns, t5_target_c)

    target_capacity_w = startup_cfg.get("target_vcc_cooling_capacity_w")
    target_m_air = startup_cfg.get("target_air_mass_flow_kg_s")
    if target_capacity_w is not None:
        iteration_count += _solve_air_speed_for_evaporator_capacity(config, model, startup_cfg, unknowns, float(target_capacity_w))
    elif target_m_air is not None:
        iteration_count += _solve_air_speed_for_mass_flow(config, model, startup_cfg, unknowns, float(target_m_air))

    room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, _, dock_c = unknowns
    air = model._evaluate_air_cycle(room_c + KELVIN_OFFSET, t3_c + KELVIN_OFFSET, t4_c + KELVIN_OFFSET, t6_c + KELVIN_OFFSET)
    q_dock = model._evaluate_dock_evaporator(dock_c, tevap_c)
    q_evap_total = air["q_cascade"] + q_dock

    ref_fluid = config["fluids"]["refrigerant"]
    vcc_cfg = config["vcc_cycle"]
    p_evap = p_sat(tevap_c + KELVIN_OFFSET, ref_fluid)
    p_cond = p_sat(tcond_c + KELVIN_OFFSET, ref_fluid)
    h9 = h_refrigerant_liquid(tcond_c + KELVIN_OFFSET - vcc_cfg["subcooling_k"], p_cond, ref_fluid, vcc_cfg["subcooling_k"])
    superheat_target_k = float(startup_cfg.get("superheat_target_k", 5.0))
    if startup_cfg.get("solve_refrigerant_compressor_speed", True):
        h7_target = float(PropsSI("H", "T", tevap_c + KELVIN_OFFSET + superheat_target_k, "P", p_evap, ref_fluid))
        m_ref = q_evap_total / max(h7_target - h9, 1.0e-9)
        unknowns[STATE_INDEX["m_ref_kg_s"]] = m_ref
        iteration_count += _solve_refrigerant_speed_for_mass_flow(config, startup_cfg, tevap_c + KELVIN_OFFSET, tcond_c + KELVIN_OFFSET, m_ref)
    else:
        compressor_cfg = vcc_cfg["compressor"]
        m_ref = ammonia_compressor_map(
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            compressor_cfg["speed_rpm"],
            check_range=compressor_cfg.get("check_range", True),
        )["mdot_kg_s"]
        unknowns[STATE_INDEX["m_ref_kg_s"]] = m_ref

    valve_cfg = vcc_cfg["expansion_valve"]
    valve_opening = m_ref / max(valve_cfg["flow_coefficient_kg_s_pa"] * max(p_cond - p_evap, 0.0), 1.0e-12)
    valve_cfg["opening"] = float(np.clip(valve_opening, valve_cfg.get("opening_min", 0.05), valve_cfg.get("opening_max", 1.0)))

    ref = model._evaluate_refrigerant_cycle(tevap_c + KELVIN_OFFSET, tcond_c + KELVIN_OFFSET, m_ref, air["q_cascade"], q_dock)
    reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
    cascade_lmtd = positive_lmtd(air["t2_k"] - KELVIN_OFFSET - tevap_c, t3_c - tevap_c)
    condenser_lmtd = positive_lmtd(tcond_c - config["boundary_conditions"]["ambient_c"], tcond_c - sink_c)

    config["air_cycle"]["regenerator_ua_w_k"] = air["q_reg_hot"] / max(reg_lmtd, 1.0e-9)
    config["vcc_cycle"]["cascade_ua_w_k"] = air["q_cascade"] / max(cascade_lmtd, 1.0e-9)
    config["vcc_cycle"]["condenser_ua_w_k"] = ref["q_cond"] / max(condenser_lmtd, 1.0e-9)
    _set_and_mirror_room_load(config, air["q_room"])

    sink_delta_t_c = sink_c - config["boundary_conditions"]["ambient_c"]
    if abs(sink_delta_t_c) > 1.0e-9:
        config["boundary_conditions"]["sink_m_dot_kg_s"] = ref["q_cond"] / (config["boundary_conditions"]["sink_cp_j_kg_k"] * sink_delta_t_c)

    snapshot = model.startup_metrics(unknowns, time_s)
    free_parameter_values = {
        path: float(get_path(config, path))
        for path in PAPER_DESIGN_SOLVED_PATHS
        if path != "air_cycle.compressor_mass_flow.speed_rpm" or target_m_air is not None or target_capacity_w is not None
    }
    for path, value in free_parameter_values.items():
        snapshot[f"startup_solved_{path.replace('.', '_')}"] = value
    snapshot["startup_paper_t5_target_c"] = t5_target_c
    snapshot["startup_paper_regenerator_effectiveness"] = float(startup_cfg.get("regenerator_effectiveness", 0.9))
    if target_capacity_w is not None:
        snapshot["startup_paper_target_vcc_cooling_capacity_w"] = float(target_capacity_w)

    return StartupInitializationResult(
        unknowns=unknowns,
        iterations=iteration_count,
        cost=0.0,
        snapshot=snapshot,
        cache_hit=False,
    )


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

    if mode == "paper_design_point":
        result = _solve_paper_design_initialization(config, model, startup_cfg, time_s)
        if use_cache:
            free_parameter_values = {
                path: float(get_path(config, path))
                for path in PAPER_DESIGN_SOLVED_PATHS
                if f"startup_solved_{path.replace('.', '_')}" in result.snapshot
            }
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
            f"[startup] solved paper_design_point | iterations={result.iterations} | cost={result.cost:.3e} | "
            f"room={result.snapshot['room_c']:.2f} C | t5={result.snapshot['t5_c']:.2f} C | "
            f"m_air={result.snapshot['m_air_kg_s']:.4f} kg/s"
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
        scaled = []
        for idx, value in enumerate(balance):
            scale = scales["kg_s"] if idx >= len(balance) - 2 else scales["w"]
            scaled.append(value / scale)
        return np.asarray(scaled, dtype=float)

    room_delta_t_target_c = startup_cfg.get("room_delta_t_target_c")
    t5_target_c = startup_cfg.get("t5_target_c")
    superheat_target_k = startup_cfg.get("superheat_target_k")
    startup_residual_size = len(model.startup_balance_residual(fixed_unknowns, time_s))
    if room_delta_t_target_c is not None:
        startup_residual_size += 1
    if t5_target_c is not None:
        startup_residual_size += 1
    if superheat_target_k is not None:
        startup_residual_size += 1

    def startup_residual_fn(x: np.ndarray) -> np.ndarray:
        try:
            residual = scaled_balance_residual_fn(x)
        except ValueError:
            return np.full(startup_residual_size, 1.0e6, dtype=float)
        if room_delta_t_target_c is None and t5_target_c is None and superheat_target_k is None:
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
        if superheat_target_k is not None:
            extra_residuals.append((metrics["refrigerant_superheat_k"] - float(superheat_target_k)) / scales["delta_t_c"])
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
    state = np.array([unknowns[0], unknowns[1], unknowns[8]], dtype=float)
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
    dock_c = np.array([row["dock_c"] for row in history])
    t3_c = np.array([row["t3_c"] for row in history])
    t5_c = np.array([row["t5_c"] for row in history])
    m_ref_kg_s = np.array([row["m_ref_kg_s"] for row in history])
    m_air_kg_s = np.array([row["m_air_kg_s"] for row in history])
    cop = np.array([row["cop_system"] for row in history])

    fig, axes = plt.subplots(7, 1, figsize=(10, 17), sharex=True)
    for ax in axes:
        ax.ticklabel_format(axis="y", useOffset=False)

    axes[0].plot(t_min, room_c, label="Room")
    axes[0].set_ylabel("Temperature [C]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, dock_c, label="Loading dock")
    axes[1].set_ylabel("Temperature [C]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_min, t3_c, label="Air after cascade exchanger")
    axes[2].set_ylabel("Temperature [C]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_min, t5_c, label="Air entering room")
    axes[3].set_ylabel("Temperature [C]")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t_min, m_ref_kg_s, label="Refrigerant mass flow")
    axes[4].set_ylabel("Mass Flow [kg/s]")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(t_min, m_air_kg_s, label="Air mass flow")
    axes[5].set_ylabel("Mass Flow [kg/s]")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(t_min, cop, label="COP")
    axes[6].set_xlabel("Time [min]")
    axes[6].set_ylabel("COP")
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

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
