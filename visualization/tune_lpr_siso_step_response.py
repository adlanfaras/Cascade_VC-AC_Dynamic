from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from src.cascade_dynamics.config import load_config
from src.cascade_dynamics.fluids import DEFAULT_REFPROP_PATH
from src.cascade_dynamics.simulation import run_simulation, save_csv


AIR_SPEED_PATH = "air_cycle.compressor_mass_flow.speed_rpm"
NH3_SPEED_PATH = "vcc_cycle.compressor.speed_rpm"
AIR_ROOM_CONTROLLER = "B1_air_before_room_to_air_compressor_speed"
NH3_ROOM_CONTROLLER = "B2_air_after_cascade_to_nh3_compressor_speed"

ACTUATOR_PATH = AIR_SPEED_PATH
ROOM_CONTROLLER = AIR_ROOM_CONTROLLER
STARTUP_ACTUATOR_KEY = "startup_solved_air_cycle_compressor_mass_flow_speed_rpm"
OPEN_LOOP_CONTROLLER_NAME = "open_loop_air_speed_step"
ACTUATOR_NAME = "air compressor"
ACTUATOR_TITLE = "Air Compressor"
ACTUATOR_SYMBOL = "N_air"
ACTUATOR_COLUMN_PREFIX = "air"
AUX_RESPONSE_KEY = "m_air_kg_s"
AUX_RESPONSE_SYMBOL = "m_dot_air"
ROOM_CONTROL_ACTION = "direct"
ACTUATOR_BIAS_RPM: float | None = None
ACTUATOR_U_MIN_RPM: float | None = None
ACTUATOR_U_MAX_RPM: float | None = None
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


@dataclass(frozen=True)
class StepModel:
    config_name: str
    baseline_speed_rpm: float
    step_speed_rpm: float
    step_delta_rpm: float
    y0_c: float
    y_final_c: float
    process_gain_c_per_rpm: float
    tau_s: float
    theta_s: float
    tangent_slope_c_per_s: float
    tangent_initial_intercept_s: float
    tangent_final_intercept_s: float
    lambda_s: float
    pi_gain_rpm_per_k: float
    pi_ti_min: float
    fit_rmse_k: float
    fit_r2: float
    clipped_gain: bool
    clipped_ti: bool


def _set_path(data: dict[str, Any], path: str, value: float) -> None:
    current: Any = data
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = float(value)


def configure_actuator_experiment(actuator: str) -> None:
    global ACTUATOR_PATH
    global ROOM_CONTROLLER
    global STARTUP_ACTUATOR_KEY
    global OPEN_LOOP_CONTROLLER_NAME
    global ACTUATOR_NAME
    global ACTUATOR_TITLE
    global ACTUATOR_SYMBOL
    global ACTUATOR_COLUMN_PREFIX
    global AUX_RESPONSE_KEY
    global AUX_RESPONSE_SYMBOL
    global ROOM_CONTROL_ACTION
    global ACTUATOR_BIAS_RPM
    global ACTUATOR_U_MIN_RPM
    global ACTUATOR_U_MAX_RPM

    if actuator == "air-compressor":
        ACTUATOR_PATH = AIR_SPEED_PATH
        ROOM_CONTROLLER = AIR_ROOM_CONTROLLER
        STARTUP_ACTUATOR_KEY = "startup_solved_air_cycle_compressor_mass_flow_speed_rpm"
        OPEN_LOOP_CONTROLLER_NAME = "open_loop_air_speed_step"
        ACTUATOR_NAME = "air compressor"
        ACTUATOR_TITLE = "Air Compressor"
        ACTUATOR_SYMBOL = "N_air"
        ACTUATOR_COLUMN_PREFIX = "air"
        AUX_RESPONSE_KEY = "m_air_kg_s"
        AUX_RESPONSE_SYMBOL = "m_dot_air"
        ROOM_CONTROL_ACTION = "direct"
        ACTUATOR_BIAS_RPM = None
        ACTUATOR_U_MIN_RPM = 10000.0
        ACTUATOR_U_MAX_RPM = 20000.0
    elif actuator == "nh3-compressor":
        ACTUATOR_PATH = NH3_SPEED_PATH
        ROOM_CONTROLLER = NH3_ROOM_CONTROLLER
        STARTUP_ACTUATOR_KEY = "startup_solved_vcc_cycle_compressor_speed_rpm"
        OPEN_LOOP_CONTROLLER_NAME = "open_loop_nh3_speed_step"
        ACTUATOR_NAME = "NH3 compressor"
        ACTUATOR_TITLE = "NH3 Compressor"
        ACTUATOR_SYMBOL = "N_NH3"
        ACTUATOR_COLUMN_PREFIX = "nh3"
        AUX_RESPONSE_KEY = "m_ref_kg_s"
        AUX_RESPONSE_SYMBOL = "m_dot_NH3"
        ROOM_CONTROL_ACTION = "direct"
        ACTUATOR_BIAS_RPM = 1450.0
        ACTUATOR_U_MIN_RPM = 1226.0
        ACTUATOR_U_MAX_RPM = 20000.0
    else:
        raise ValueError(f"Unsupported actuator experiment: {actuator}")


def _series(history: list[dict[str, float]], key: str, default: float = np.nan) -> np.ndarray:
    return np.array([float(row.get(key, default)) for row in history], dtype=float)


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 22,
            "axes.labelsize": 28,
            "axes.titlesize": 24,
            "axes.linewidth": 1.6,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 18,
            "legend.title_fontsize": 22,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", direction="in", top=True, right=True, length=8, width=1.4)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=4, width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def _controller(config: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    controller_name = ROOM_CONTROLLER if name is None else name
    for controller in config.get("control", {}).get("controllers", []):
        if controller.get("name") == controller_name:
            return controller
    raise KeyError(f"Controller {controller_name!r} was not found")


def _force_siso_room_control(config: dict[str, Any]) -> None:
    config.setdefault("control", {})["enabled"] = True
    for controller in config["control"].get("controllers", []):
        controller["enabled"] = controller.get("name") == ROOM_CONTROLLER
    room = _controller(config)
    room.update(
        {
            "measurement": "room_c",
            "setpoint": -30.0,
            "setpoint_from_startup": "room_c",
            "actuator_path": ACTUATOR_PATH,
            "action": ROOM_CONTROL_ACTION,
            "anti_windup": True,
        }
    )
    if ACTUATOR_BIAS_RPM is not None:
        room["bias"] = float(ACTUATOR_BIAS_RPM)
    if ACTUATOR_U_MIN_RPM is not None:
        room["u_min"] = float(ACTUATOR_U_MIN_RPM)
    if ACTUATOR_U_MAX_RPM is not None:
        room["u_max"] = float(ACTUATOR_U_MAX_RPM)


def _disable_disturbances(config: dict[str, Any]) -> None:
    infiltration = config.setdefault("disturbances", {}).setdefault("infiltration", {})
    infiltration["enabled"] = False
    boundary = config.setdefault("boundary_conditions", {})
    if "load_before_w" in boundary:
        boundary["load_after_w"] = float(boundary["load_before_w"])
    if "dock_load_before_w" in boundary:
        boundary["dock_load_after_w"] = float(boundary["dock_load_before_w"])


def _remove_startup_free_parameter(config: dict[str, Any], path: str) -> None:
    startup = config.setdefault("simulation", {}).setdefault("startup_initialization", {})
    free_parameters = startup.get("free_parameters")
    if not isinstance(free_parameters, list):
        return
    startup["free_parameters"] = [item for item in free_parameters if item.get("path") != path]


def _quiet_simulation_config(config: dict[str, Any], *, t_end_s: float | None = None, dt_s: float | None = None) -> None:
    sim = config.setdefault("simulation", {})
    if t_end_s is not None:
        sim["t_end_s"] = float(t_end_s)
    if dt_s is not None:
        sim["dt_s"] = float(dt_s)
    sim["progress_interval_steps"] = max(int(sim.get("progress_interval_steps", 999999)), 999999)


def _with_backend(config: dict[str, Any], backend: str, refprop_path: str | None) -> None:
    fluids = config.setdefault("fluids", {})
    fluids["property_backend"] = backend
    if backend == "refprop" and refprop_path:
        fluids["refprop_path"] = refprop_path


def _baseline_startup_speed(config: dict[str, Any], backend: str, refprop_path: str | None) -> float:
    startup_config = deepcopy(config)
    _force_siso_room_control(startup_config)
    _disable_disturbances(startup_config)
    _quiet_simulation_config(startup_config, t_end_s=0.0)
    _with_backend(startup_config, backend, refprop_path)
    history = run_simulation(startup_config)
    first = history[0]
    if STARTUP_ACTUATOR_KEY in first and math.isfinite(float(first[STARTUP_ACTUATOR_KEY])):
        return float(first[STARTUP_ACTUATOR_KEY])
    if ACTUATOR_PATH == AIR_SPEED_PATH and "air_compressor_map_speed_eval_rpm" in first:
        return float(first["air_compressor_map_speed_eval_rpm"])
    current: Any = startup_config
    for part in ACTUATOR_PATH.split("."):
        current = current[part]
    return float(current)


def _make_step_config(
    config: dict[str, Any],
    output_dir: Path,
    baseline_speed_rpm: float,
    step_delta_rpm: float,
    step_time_s: float,
    backend: str,
    refprop_path: str | None,
    dt_s: float,
    t_end_s: float | None,
) -> tuple[dict[str, Any], float]:
    step_config = deepcopy(config)
    _disable_disturbances(step_config)
    _quiet_simulation_config(step_config, t_end_s=t_end_s, dt_s=dt_s)
    _with_backend(step_config, backend, refprop_path)

    room = _controller(step_config)
    u_min = float(room.get("u_min", 10000.0))
    u_max = float(room.get("u_max", 18000.0))
    step_speed_rpm = min(max(baseline_speed_rpm + step_delta_rpm, u_min), u_max)
    actual_delta = step_speed_rpm - baseline_speed_rpm
    if abs(actual_delta) < 1.0:
        raise ValueError(f"{ACTUATOR_NAME} speed step is too small after actuator limits")

    step_config["control"] = {
        "enabled": True,
        "controllers": [
            {
                "name": OPEN_LOOP_CONTROLLER_NAME,
                "enabled": True,
                "measurement": "time_s",
                "setpoint": float(step_time_s),
                "actuator_path": ACTUATOR_PATH,
                "action": "direct",
                "bias": baseline_speed_rpm,
                "bias_from_startup": False,
                "gain": abs(actual_delta) * 20.0,
                "Ti_min": 1.0e9,
                "u_min": min(baseline_speed_rpm, step_speed_rpm),
                "u_max": max(baseline_speed_rpm, step_speed_rpm),
                "anti_windup": False,
            }
        ],
    }

    stem = Path(config["output"]["csv_file"]).stem
    step_config.setdefault("output", {})["csv_file"] = str(output_dir / f"{stem}_open_loop_step.csv")
    step_config["output"]["plot_file"] = str(output_dir / f"{stem}_open_loop_step.png")
    return step_config, actual_delta


def _fopdt_response(
    t_s: np.ndarray,
    y0: float,
    process_gain: float,
    step_delta: float,
    tau_s: float,
    theta_s: float,
    step_time_s: float,
) -> np.ndarray:
    elapsed = t_s - step_time_s - theta_s
    fraction = np.where(elapsed > 0.0, 1.0 - np.exp(-elapsed / max(tau_s, 1.0e-9)), 0.0)
    return y0 + process_gain * step_delta * fraction


def _zn_tangent_line(t_s: np.ndarray, slope: float, point_t_s: float, point_y: float) -> np.ndarray:
    return point_y + slope * (t_s - point_t_s)


def _identify_zn_open_loop(
    history: list[dict[str, float]],
    config_name: str,
    baseline_speed_rpm: float,
    step_delta_rpm: float,
    step_time_s: float,
    *,
    gain_min: float,
    gain_max: float,
    ti_min_min: float,
    ti_min_max: float,
    theta_floor_s: float,
) -> StepModel:
    t_s = _series(history, "time_s", 0.0)
    y = _series(history, "room_c")
    speed = _series(history, f"pid_{OPEN_LOOP_CONTROLLER_NAME}_output", baseline_speed_rpm)
    if t_s.size < 4:
        raise ValueError("Step history does not contain enough pre/post-step samples")

    threshold = max(1.0, 0.5 * abs(step_delta_rpm))
    changed = np.nonzero((t_s >= step_time_s) & (np.abs(speed - baseline_speed_rpm) >= threshold))[0]
    if changed.size:
        step_index = max(int(changed[0]) - 1, 0)
    else:
        candidates = np.nonzero(t_s >= step_time_s)[0]
        step_index = int(candidates[0]) if candidates.size else 0

    y0 = float(y[step_index])
    tail_count = max(3, int(0.10 * max(t_s.size - step_index, 1)))
    y_final = float(np.nanmean(y[-tail_count:]))
    process_gain = (y_final - y0) / step_delta_rpm
    if abs(process_gain) < 1.0e-9:
        raise ValueError("Room temperature response is too small for step identification")

    direction = 1.0 if y_final > y0 else -1.0
    segment_t0 = t_s[step_index:-1]
    segment_t1 = t_s[step_index + 1 :]
    segment_y0 = y[step_index:-1]
    segment_y1 = y[step_index + 1 :]
    dt = segment_t1 - segment_t0
    slopes = np.divide(segment_y1 - segment_y0, dt, out=np.full_like(dt, np.nan, dtype=float), where=dt > 0.0)
    directional_slopes = direction * slopes
    valid = np.isfinite(directional_slopes) & (directional_slopes > 0.0)
    if not np.any(valid):
        raise ValueError("Could not find a usable reaction-curve tangent segment")

    valid_indices = np.nonzero(valid)[0]
    tangent_segment = int(valid_indices[np.argmax(directional_slopes[valid])])
    slope = float(slopes[tangent_segment])
    point_t = float(0.5 * (segment_t0[tangent_segment] + segment_t1[tangent_segment]))
    point_y = float(0.5 * (segment_y0[tangent_segment] + segment_y1[tangent_segment]))
    t_initial = point_t + (y0 - point_y) / slope
    t_final = point_t + (y_final - point_y) / slope
    if t_final < t_initial:
        t_initial, t_final = t_final, t_initial

    theta_s = max(float(t_initial - step_time_s), theta_floor_s)
    tau_s = max(float(t_final - t_initial), 1.0)

    # Open-loop Ziegler-Nichols process-reaction-curve PI rule.
    raw_gain = 0.9 * tau_s / (abs(process_gain) * theta_s)
    raw_ti_min = 3.33 * theta_s / 60.0
    tuned_gain = float(np.clip(raw_gain, gain_min, gain_max))
    tuned_ti_min = float(np.clip(raw_ti_min, ti_min_min, ti_min_max))

    return StepModel(
        config_name=config_name,
        baseline_speed_rpm=baseline_speed_rpm,
        step_speed_rpm=baseline_speed_rpm + step_delta_rpm,
        step_delta_rpm=step_delta_rpm,
        y0_c=y0,
        y_final_c=y_final,
        process_gain_c_per_rpm=process_gain,
        tau_s=tau_s,
        theta_s=theta_s,
        tangent_slope_c_per_s=slope,
        tangent_initial_intercept_s=t_initial,
        tangent_final_intercept_s=t_final,
        lambda_s=np.nan,
        pi_gain_rpm_per_k=tuned_gain,
        pi_ti_min=tuned_ti_min,
        fit_rmse_k=np.nan,
        fit_r2=np.nan,
        clipped_gain=not math.isclose(raw_gain, tuned_gain, rel_tol=1.0e-9, abs_tol=1.0e-9),
        clipped_ti=not math.isclose(raw_ti_min, tuned_ti_min, rel_tol=1.0e-9, abs_tol=1.0e-9),
    )


def _apply_tuning(config_path: Path, model: StepModel) -> None:
    config = load_config(config_path)
    _force_siso_room_control(config)
    room = _controller(config)
    room["gain"] = round(model.pi_gain_rpm_per_k, 6)
    room["Ti_min"] = round(model.pi_ti_min, 6)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def _run_closed_loop_verification(
    config_path: Path,
    output_dir: Path,
    backend: str,
    refprop_path: str | None,
    dt_s: float,
) -> list[dict[str, float]]:
    config = load_config(config_path)
    _force_siso_room_control(config)
    _quiet_simulation_config(config, dt_s=dt_s)
    _with_backend(config, backend, refprop_path)
    stem = config_path.stem
    config.setdefault("output", {})["csv_file"] = str(output_dir / f"{stem}_closed_loop_tuned.csv")
    config["output"]["plot_file"] = str(output_dir / f"{stem}_closed_loop_tuned.png")
    history = run_simulation(config)
    save_csv(history, config["output"]["csv_file"])
    return history


def _make_pi_load_step_config(
    config_path: Path,
    output_dir: Path,
    backend: str,
    refprop_path: str | None,
    dt_s: float,
    t_end_s: float,
    step_time_s: float,
    load_step_fraction: float,
) -> dict[str, Any]:
    config = load_config(config_path)
    _force_siso_room_control(config)
    _disable_disturbances(config)
    _quiet_simulation_config(config, t_end_s=t_end_s, dt_s=dt_s)
    _with_backend(config, backend, refprop_path)

    boundary = config.setdefault("boundary_conditions", {})
    load_before = float(boundary["load_before_w"])
    boundary["load_step_time_s"] = float(step_time_s)
    boundary["load_after_w"] = load_before * (1.0 + load_step_fraction)

    stem = config_path.stem
    output_cfg = config.setdefault("output", {})
    output_cfg["csv_file"] = str(output_dir / f"{stem}_pi_room_load_step.csv")
    output_cfg["plot_file"] = str(output_dir / f"{stem}_pi_room_load_step.png")
    return config


def _plot_pi_load_step_performance(
    history: list[dict[str, float]],
    output_path: Path,
    title: str,
    step_time_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_s = _series(history, "time_s", 0.0)
    t_min = (t_s - step_time_s) / 60.0
    load_w = _series(history, "load_w")
    q_room_w = _series(history, "q_room_w")
    room_c = _series(history, "room_c")
    speed = _series(history, f"pid_{ROOM_CONTROLLER}_output")
    setpoint = float(np.nanmedian(room_c[t_s < step_time_s])) if np.any(t_s < step_time_s) else -30.0

    pre_mask = t_s < step_time_s
    load_ref = float(np.nanmean(load_w[pre_mask])) if np.any(pre_mask) else float(load_w[0])
    speed_ref = float(np.nanmean(speed[pre_mask])) if np.any(pre_mask) else float(speed[0])

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 13.0), sharex=True)
    fig.suptitle(f"Performance of PI control in response to a room-load step: {title}", y=0.995)

    axes[0].plot(t_min, load_w / 1000.0, color=OKABE_ITO["black"], linewidth=2.0, label=r"$Q_{load}$")
    axes[0].plot(t_min, q_room_w / 1000.0, color=OKABE_ITO["vermillion"], linewidth=2.8, label=r"$Q_e$")
    axes[0].axvline(0.0, color="0.25", linewidth=1.5, linestyle=":")
    axes[0].set_ylabel(r"$Q$ (kW)")
    axes[0].legend(frameon=False, loc="best", ncols=2)
    axes[0].text(0.02, 0.08, "(a)", transform=axes[0].transAxes, fontsize=20)

    axes[1].axhline(setpoint, color=OKABE_ITO["black"], linewidth=2.0, label=r"$T_{r,set}$")
    axes[1].plot(t_min, room_c, color=OKABE_ITO["vermillion"], linewidth=2.8, label=r"$T_r$")
    axes[1].axvline(0.0, color="0.25", linewidth=1.5, linestyle=":")
    axes[1].set_ylabel(r"$T_{room}$ ($^\circ$C)")
    axes[1].legend(frameon=False, loc="best", ncols=2)
    axes[1].text(0.02, 0.08, "(b)", transform=axes[1].transAxes, fontsize=20)

    axes[2].axhline(speed_ref, color=OKABE_ITO["black"], linewidth=2.0, label=rf"${ACTUATOR_SYMBOL},0$")
    axes[2].plot(t_min, speed, color=OKABE_ITO["vermillion"], linewidth=2.8, label=rf"${ACTUATOR_SYMBOL}$")
    axes[2].axvline(0.0, color="0.25", linewidth=1.5, linestyle=":")
    axes[2].set_ylabel(rf"${ACTUATOR_SYMBOL}$ (rpm)")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend(frameon=False, loc="best", ncols=2)
    axes[2].text(0.02, 0.08, "(c)", transform=axes[2].transAxes, fontsize=20)

    for ax in axes:
        style_axes(ax)
        ax.set_xlim(-3.0, max(15.0, float(np.nanmax(t_min))))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_pi_load_step_case(
    config_path: Path,
    output_dir: Path,
    backend: str,
    refprop_path: str | None,
    dt_s: float,
    t_end_s: float,
    step_time_s: float,
    load_step_fraction: float,
) -> tuple[list[dict[str, float]], Path]:
    config = _make_pi_load_step_config(
        config_path,
        output_dir,
        backend,
        refprop_path,
        dt_s,
        t_end_s,
        step_time_s,
        load_step_fraction,
    )
    history = run_simulation(config)
    save_csv(history, config["output"]["csv_file"])
    plot_path = Path(config["output"]["plot_file"])
    _plot_pi_load_step_performance(history, plot_path, config_path.name, step_time_s)
    return history, plot_path


def _plot_identification(
    history: list[dict[str, float]],
    model: StepModel,
    output_path: Path,
    step_time_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_s = _series(history, "time_s", 0.0)
    room_c = _series(history, "room_c")
    speed = _series(history, f"pid_{OPEN_LOOP_CONTROLLER_NAME}_output", model.baseline_speed_rpm)
    tangent = _zn_tangent_line(
        t_s,
        model.tangent_slope_c_per_s,
        0.5 * (model.tangent_initial_intercept_s + model.tangent_final_intercept_s),
        0.5 * (model.y0_c + model.y_final_c),
    )
    tangent_mask = (t_s >= model.tangent_initial_intercept_s - 60.0) & (
        t_s <= model.tangent_final_intercept_s + 60.0
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    axes[0].plot(t_s / 60.0, room_c, label="simulated room")
    axes[0].plot(t_s[tangent_mask] / 60.0, tangent[tangent_mask], "--", label="ZN tangent")
    axes[0].axhline(model.y0_c, color="tab:gray", linestyle=":", linewidth=1.0, label="initial/final levels")
    axes[0].axhline(model.y_final_c, color="tab:gray", linestyle=":", linewidth=1.0)
    axes[0].axvline(model.tangent_initial_intercept_s / 60.0, color="tab:purple", linestyle=":", linewidth=1.0)
    axes[0].axvline(model.tangent_final_intercept_s / 60.0, color="tab:purple", linestyle=":", linewidth=1.0)
    axes[0].axvline(step_time_s / 60.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Room [C]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_s / 60.0, speed, color="tab:orange", label=f"{ACTUATOR_NAME} speed")
    axes[1].axvline(step_time_s / 60.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Speed [rpm]")
    axes[1].set_xlabel("Time [min]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(model.config_name)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_reaction_curve_panels(
    history: list[dict[str, float]],
    model: StepModel,
    output_path: Path,
    step_time_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_s = _series(history, "time_s", 0.0)
    t_min = (t_s - step_time_s) / 60.0
    speed = _series(history, f"pid_{OPEN_LOOP_CONTROLLER_NAME}_output", model.baseline_speed_rpm)
    room_c = _series(history, "room_c")
    aux_response = _series(history, AUX_RESPONSE_KEY)
    tangent = _zn_tangent_line(
        t_s,
        model.tangent_slope_c_per_s,
        0.5 * (model.tangent_initial_intercept_s + model.tangent_final_intercept_s),
        0.5 * (model.y0_c + model.y_final_c),
    )
    tangent_mask = (t_s >= model.tangent_initial_intercept_s - 60.0) & (
        t_s <= model.tangent_final_intercept_s + 60.0
    )

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 13.0), sharex=True)
    fig.suptitle(f"Open-loop Ziegler-Nichols step test: {model.config_name}", y=0.995)

    axes[0].plot(t_min, np.full_like(t_min, model.baseline_speed_rpm), color=OKABE_ITO["black"], linewidth=2.0, label=rf"${ACTUATOR_SYMBOL},0$")
    axes[0].plot(t_min, speed, color=OKABE_ITO["vermillion"], linewidth=2.8, label=rf"${ACTUATOR_SYMBOL}$")
    axes[0].set_ylabel(rf"${ACTUATOR_SYMBOL}$ (rpm)")
    axes[0].legend(frameon=False, loc="best", ncols=2)
    axes[0].text(0.02, 0.08, "(a)", transform=axes[0].transAxes, fontsize=20)

    axes[1].axhline(model.y0_c, color=OKABE_ITO["black"], linewidth=2.0, label=r"$T_{r,0}$")
    axes[1].axhline(model.y_final_c, color="0.4", linewidth=1.8, linestyle=":", label=r"$T_{r,\infty}$")
    axes[1].plot(t_min, room_c, color=OKABE_ITO["vermillion"], linewidth=2.8, label=r"$T_r$")
    axes[1].plot(t_min[tangent_mask], tangent[tangent_mask], color=OKABE_ITO["blue"], linewidth=2.2, linestyle="--", label="ZN tangent")
    step_x = 0.0
    delay_x = (model.tangent_initial_intercept_s - step_time_s) / 60.0
    final_x = (model.tangent_final_intercept_s - step_time_s) / 60.0
    axes[1].axvline(step_x, color=OKABE_ITO["black"], linewidth=1.6, linestyle=":")
    axes[1].axvline(delay_x, color=OKABE_ITO["purple"], linewidth=1.6, linestyle=":")
    axes[1].axvline(final_x, color=OKABE_ITO["purple"], linewidth=1.6, linestyle=":")
    axes[1].set_ylabel(r"$T_{room}$ ($^\circ$C)")
    axes[1].legend(frameon=False, loc="best", ncols=2)
    axes[1].text(0.02, 0.08, "(b)", transform=axes[1].transAxes, fontsize=20)

    y_arrow = model.y0_c + 0.30 * (model.y_final_c - model.y0_c)
    y_offset = 0.08 * abs(model.y_final_c - model.y0_c)
    axes[1].annotate(
        "",
        xy=(step_x, y_arrow),
        xytext=(delay_x, y_arrow),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.4, "color": OKABE_ITO["black"]},
    )
    axes[1].text(0.5 * (step_x + delay_x), y_arrow + y_offset, r"$L$", ha="center", va="bottom")
    axes[1].annotate(
        "",
        xy=(delay_x, y_arrow),
        xytext=(final_x, y_arrow),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.4, "color": OKABE_ITO["black"]},
    )
    axes[1].text(0.5 * (delay_x + final_x), y_arrow - y_offset, r"$T$", ha="center", va="top")

    if not np.all(np.isnan(aux_response)):
        m0 = float(aux_response[np.nonzero(t_s >= step_time_s)[0][0]]) if np.any(t_s >= step_time_s) else float(aux_response[0])
        axes[2].axhline(m0, color=OKABE_ITO["black"], linewidth=2.0, label=rf"$\dot{{{AUX_RESPONSE_SYMBOL}}},0$")
        axes[2].plot(t_min, aux_response, color=OKABE_ITO["vermillion"], linewidth=2.8, label=rf"$\dot{{{AUX_RESPONSE_SYMBOL}}}$")
    else:
        axes[2].plot(t_min, speed, color=OKABE_ITO["vermillion"], linewidth=2.8, label=rf"${ACTUATOR_SYMBOL}$")
    axes[2].set_ylabel(rf"$\dot{{{AUX_RESPONSE_SYMBOL}}}$ (kg/s)")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend(frameon=False, loc="best", ncols=2)
    axes[2].text(0.02, 0.08, "(c)", transform=axes[2].transAxes, fontsize=20)

    for ax in axes:
        style_axes(ax)
        ax.set_xlim(-3.0, max(15.0, final_x + 2.0))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _closed_loop_metrics(history: list[dict[str, float]]) -> dict[str, float]:
    t_s = _series(history, "time_s", 0.0)
    room_c = _series(history, "room_c")
    speed = _series(history, f"pid_{ROOM_CONTROLLER}_output")
    power_w = _series(history, "w_total_input_w", 0.0)
    setpoint = -30.0
    return {
        "initial_room_c": float(room_c[0]),
        "final_room_c": float(room_c[-1]),
        "min_room_c": float(np.nanmin(room_c)),
        "max_room_c": float(np.nanmax(room_c)),
        "max_room_excursion_k": float(np.nanmax(room_c - setpoint)),
        f"min_{ACTUATOR_COLUMN_PREFIX}_speed_rpm": float(np.nanmin(speed)),
        f"max_{ACTUATOR_COLUMN_PREFIX}_speed_rpm": float(np.nanmax(speed)),
        f"final_{ACTUATOR_COLUMN_PREFIX}_speed_rpm": float(speed[-1]),
        "energy_kwh": float(np.trapezoid(power_w, t_s) / 3.6e6) if t_s.size > 1 else 0.0,
    }


def _empty_closed_loop_metrics() -> dict[str, float]:
    return {
        "initial_room_c": np.nan,
        "final_room_c": np.nan,
        "min_room_c": np.nan,
        "max_room_c": np.nan,
        "max_room_excursion_k": np.nan,
        f"min_{ACTUATOR_COLUMN_PREFIX}_speed_rpm": np.nan,
        f"max_{ACTUATOR_COLUMN_PREFIX}_speed_rpm": np.nan,
        f"final_{ACTUATOR_COLUMN_PREFIX}_speed_rpm": np.nan,
        "energy_kwh": np.nan,
    }


def _plot_closed_loop(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    t_s = _series(history, "time_s", 0.0)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 9.0), sharex=True)
    axes[0].plot(t_s / 60.0, _series(history, "room_c"), label="room")
    axes[0].axhline(-30.0, color="black", linestyle=":", linewidth=1.0, label="setpoint")
    axes[1].plot(t_s / 60.0, _series(history, f"pid_{ROOM_CONTROLLER}_output"), color="tab:orange", label=f"{ACTUATOR_NAME} speed")
    axes[2].plot(t_s / 60.0, _series(history, "infiltration_door_open_fraction", 0.0), color="tab:red", label="door")
    axes[3].plot(t_s / 60.0, _series(history, "w_total_input_w", 0.0) / 1000.0, color="tab:green", label="power")
    axes[0].set_ylabel("Room [C]")
    axes[1].set_ylabel(f"{ACTUATOR_NAME} speed [rpm]")
    axes[2].set_ylabel("Door [-]")
    axes[3].set_ylabel("Power [kW]")
    axes[3].set_xlabel("Time [min]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# LPR SISO {ACTUATOR_TITLE} Step-Response Tuning Report",
        "",
        "## Method",
        "",
        f"Property backend: `{args.property_backend}`.",
        f"REFPROP path: `{args.refprop_path}`." if args.property_backend == "refprop" else "",
        "",
        f"1. Each `*_lpr.json` case was treated as a SISO loop: `room_c` controls `{ACTUATOR_PATH}`.",
        f"2. Door infiltration and load changes were disabled for identification so the measured response came only from a {ACTUATOR_NAME} speed step.",
        f"3. A startup solve established the nominal {ACTUATOR_NAME} speed. The transient identification run then stepped that speed upward and recorded `room_c`.",
        "4. The open-loop Ziegler-Nichols reaction-curve method was applied directly to the step response:",
        "",
        "   - draw the tangent at the steepest point of the process reaction curve",
        "   - `L` is the time from the input step to the tangent intersection with the initial temperature level",
        "   - `T` is the time between the tangent intersections with the initial and final temperature levels",
        f"   - `K = delta T_room / delta {ACTUATOR_SYMBOL}`",
        "",
        "5. The PI controller was tuned with the open-loop Ziegler-Nichols PI rule:",
        "",
        "   `Kc = 0.9 * T / (abs(K) * L)`",
        "",
        "   `Ti = 3.33 * L`",
        "",
        f"The selected room controller uses `action: {ROOM_CONTROL_ACTION}` for this actuator profile; the applied controller gain is reported as a positive magnitude.",
        "",
        "## Results",
        "",
        "| Config | Step [rpm] | K [C/rpm] | T [s] | L [s] | Tangent slope [C/s] | gain [rpm/K] | Ti [min] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {config} | {step_delta_rpm:.1f} | {process_gain_c_per_rpm:.6g} | {tau_s:.1f} | {theta_s:.1f} | "
            "{tangent_slope_c_per_s:.6g} | {pi_gain_rpm_per_k:.2f} | {pi_ti_min:.2f} |".format(**row)
        )
    closed_loop_plots = [row.get("closed_loop_plot") for row in rows if row.get("closed_loop_plot")]
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Identification plots: `{args.output_dir}` with suffix `_identification.png`.",
            f"- Publication-style Ziegler-Nichols reaction-curve panels: `{args.output_dir}` with suffix `_zn_reaction_curve.png`.",
            f"- Numeric summary: `{Path(args.output_dir) / 'lpr_siso_step_response_tuning_summary.csv'}`.",
        ]
    )
    if closed_loop_plots:
        lines.append(f"- Closed-loop verification plots: `{args.output_dir}` with suffix `_closed_loop_tuned.png`.")
    pi_load_step_plots = [row.get("pi_load_step_plot") for row in rows if row.get("pi_load_step_plot")]
    if pi_load_step_plots:
        lines.append(f"- PI load-step performance plots: `{args.output_dir}` with suffix `_pi_room_load_step.png`.")
    else:
        lines.append("- Closed-loop verification was skipped for the full matrix to avoid a long nonlinear simulation batch; the tuning evidence is the open-loop Ziegler-Nichols reaction curve.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tune_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    base_config = load_config(config_path)
    _force_siso_room_control(base_config)
    output_dir = Path(args.output_dir)
    baseline_speed = _baseline_startup_speed(base_config, args.property_backend, args.refprop_path)
    step_config, actual_delta = _make_step_config(
        base_config,
        output_dir,
        baseline_speed,
        args.step_delta_rpm,
        args.step_time_s,
        args.property_backend,
        args.refprop_path,
        args.identification_dt_s,
        args.identification_t_end_s,
    )
    step_history = run_simulation(step_config)
    save_csv(step_history, step_config["output"]["csv_file"])
    model = _identify_zn_open_loop(
        step_history,
        config_path.name,
        baseline_speed,
        actual_delta,
        args.step_time_s,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        ti_min_min=args.ti_min_min,
        ti_min_max=args.ti_min_max,
        theta_floor_s=args.theta_floor_s,
    )
    _plot_identification(step_history, model, output_dir / f"{config_path.stem}_identification.png", args.step_time_s)
    reaction_curve_plot_path = output_dir / f"{config_path.stem}_zn_reaction_curve.png"
    _plot_reaction_curve_panels(step_history, model, reaction_curve_plot_path, args.step_time_s)

    tuned_config_path = config_path
    if not args.dry_run:
        _apply_tuning(config_path, model)
    else:
        tuned_config_path = _write_temp_config(config_path, base_config, model, output_dir)
    if args.skip_verification:
        metrics = _empty_closed_loop_metrics()
        closed_loop_plot = ""
    else:
        verification_history = _run_closed_loop_verification(
            tuned_config_path,
            output_dir,
            args.property_backend,
            args.refprop_path,
            args.verification_dt_s,
        )
        closed_loop_plot_path = output_dir / f"{config_path.stem}_closed_loop_tuned.png"
        _plot_closed_loop(verification_history, closed_loop_plot_path, config_path.name)
        metrics = _closed_loop_metrics(verification_history)
        closed_loop_plot = str(closed_loop_plot_path.resolve())

    pi_load_step_plot = ""
    if args.pi_load_step:
        _, pi_plot_path = run_pi_load_step_case(
            tuned_config_path,
            output_dir,
            args.property_backend,
            args.refprop_path,
            args.pi_load_step_dt_s,
            args.pi_load_step_t_end_s,
            args.pi_load_step_time_s,
            args.pi_load_step_fraction,
        )
        pi_load_step_plot = str(pi_plot_path.resolve())

    return {
        "config": config_path.name,
        **model.__dict__,
        **metrics,
        "identification_plot": str((output_dir / f"{config_path.stem}_identification.png").resolve()),
        "reaction_curve_plot": str(reaction_curve_plot_path.resolve()),
        "closed_loop_plot": closed_loop_plot,
        "pi_load_step_plot": pi_load_step_plot,
        "step_csv": str(Path(step_config["output"]["csv_file"]).resolve()),
    }


def _write_temp_config(config_path: Path, config: dict[str, Any], model: StepModel, output_dir: Path) -> Path:
    temp = deepcopy(config)
    _force_siso_room_control(temp)
    room = _controller(temp)
    room["gain"] = round(model.pi_gain_rpm_per_k, 6)
    room["Ti_min"] = round(model.pi_ti_min, 6)
    temp_path = output_dir / f"{config_path.stem}_tuned_preview.json"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(temp, handle, indent=2)
        handle.write("\n")
    return temp_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LPR SISO room-temperature PI loops from open-loop step responses.")
    parser.add_argument(
        "--actuator",
        choices=("air-compressor", "nh3-compressor"),
        default="air-compressor",
        help="Actuator used for the SISO room-temperature experiment.",
    )
    parser.add_argument("--config-dir", default="config", help="Directory containing *_lpr.json configurations.")
    parser.add_argument("--pattern", default="*_lpr.json", help="Config glob pattern relative to --config-dir.")
    parser.add_argument("--output-dir", help="Output directory.")
    parser.add_argument("--step-time-s", type=float, default=600.0, help="Open-loop speed step time.")
    parser.add_argument("--step-delta-rpm", type=float, help="Compressor speed step size.")
    parser.add_argument("--identification-dt-s", type=float, default=5.0, help="Simulation time step for open-loop step tests.")
    parser.add_argument("--identification-t-end-s", type=float, help="Optional end time for open-loop step tests.")
    parser.add_argument("--verification-dt-s", type=float, default=5.0, help="Simulation time step for closed-loop verification runs.")
    parser.add_argument("--skip-verification", action="store_true", help="Tune from step tests without running closed-loop verification.")
    parser.add_argument("--pi-load-step", action="store_true", help="Also run a closed-loop PI response to a room-load step.")
    parser.add_argument("--pi-load-step-fraction", type=float, default=0.01, help="Fractional room-load change for PI performance plot.")
    parser.add_argument("--pi-load-step-time-s", type=float, default=600.0, help="Time of the closed-loop room-load step.")
    parser.add_argument("--pi-load-step-t-end-s", type=float, default=1800.0, help="End time for the PI load-step run.")
    parser.add_argument("--pi-load-step-dt-s", type=float, default=2.5, help="Time step for the PI load-step run.")
    parser.add_argument("--theta-floor-s", type=float, default=5.0, help="Minimum effective dead time used in tuning.")
    parser.add_argument("--gain-min", type=float, default=20.0, help="Minimum applied PI gain.")
    parser.add_argument("--gain-max", type=float, default=5000.0, help="Maximum applied PI gain.")
    parser.add_argument("--ti-min-min", type=float, default=0.5, help="Minimum applied integral time in minutes.")
    parser.add_argument("--ti-min-max", type=float, default=30.0, help="Maximum applied integral time in minutes.")
    parser.add_argument("--property-backend", choices=("coolprop", "refprop"), default="refprop")
    parser.add_argument("--refprop-path", default=DEFAULT_REFPROP_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Do not overwrite config files.")
    args = parser.parse_args()
    configure_actuator_experiment(args.actuator)
    if args.output_dir is None:
        suffix = "nh3_siso_step_response_tuning" if args.actuator == "nh3-compressor" else "lpr_siso_step_response_tuning"
        args.output_dir = str(Path("outputs") / suffix)
    if args.step_delta_rpm is None:
        args.step_delta_rpm = 100.0 if args.actuator == "nh3-compressor" else 1000.0
    configure_publication_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = sorted(Path(args.config_dir).glob(args.pattern))
    if not config_paths:
        raise SystemExit(f"No configs matched {Path(args.config_dir) / args.pattern}")

    rows = []
    for config_path in config_paths:
        print(f"[tune] {config_path}")
        rows.append(tune_config(config_path, args))

    summary_path = output_dir / "lpr_siso_step_response_tuning_summary.csv"
    report_path = output_dir / "lpr_siso_step_response_tuning_report.md"
    _write_csv(rows, summary_path)
    _write_report(rows, report_path, args)
    print(f"[done] summary: {summary_path.resolve()}")
    print(f"[done] report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
