from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .humid_air import humid_air_state, saturated_room_humidity_ratio


KELVIN_OFFSET = 273.15
G = 9.81
R_AIR = 287.05
CP_AIR = 1005.0
H_FG = 2.50e6
H_FUSION = 3.34e5


@dataclass
class TianInfiltrationState:
    velocity_m_s: float = 0.0
    density_kg_m3: float = 1.2
    cumulative_volume_m3: float = 0.0
    was_open: bool = False


def zero_infiltration_result() -> dict[str, float]:
    return {
        "room_w": 0.0,
        "dock_w": 0.0,
        "q_m3_s": 0.0,
        "q_sensible_w": 0.0,
        "q_latent_w": 0.0,
        "q_total_w": 0.0,
        "cumulative_volume_m3": 0.0,
        "velocity_m_s": 0.0,
        "region_density_kg_m3": 0.0,
        "stage": 0.0,
        "door_open_fraction": 0.0,
        "effective_length_m": 0.0,
    }


def dry_air_density(temperature_k: float, pressure_pa: float) -> float:
    return float(pressure_pa) / (R_AIR * max(float(temperature_k), 1.0))


def air_density(temperature_k: float, pressure_pa: float, humidity_ratio: float | None = None) -> float:
    if humidity_ratio is None:
        return dry_air_density(temperature_k, pressure_pa)
    return humid_air_state(temperature_k, pressure_pa, humidity_ratio).mixture_density_kg_m3


def humidity_ratio_from_config(cfg: dict[str, Any], prefix: str, temperature_k: float, pressure_pa: float) -> float | None:
    for key in (
        f"{prefix}_humidity_ratio_kg_kg_da",
        f"{prefix}_omega_kg_kg_da",
        f"omega_{prefix}_kg_kg_da",
        f"{prefix}_humidity_ratio",
        f"omega_{prefix}",
    ):
        if key in cfg:
            value = float(cfg[key])
            if value > 0.2:
                value *= 1.0e-3
            return max(value, 0.0)

    for key in (f"{prefix}_relative_humidity", f"RH_{prefix}", f"{prefix}_rh"):
        if key in cfg:
            return saturated_room_humidity_ratio(temperature_k, pressure_pa, float(cfg[key]))

    return None


def _value_from_nested(cfg: dict[str, Any], *paths: tuple[str, ...], default: float | None = None) -> float:
    for path in paths:
        node: Any = cfg
        found = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                found = False
                break
            node = node[key]
        if found:
            return float(node)
    if default is None:
        raise KeyError("Missing required infiltration configuration value.")
    return float(default)


def _door_events(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    schedule = cfg.get("schedule", {})
    raw_events = schedule.get("events", cfg.get("events"))
    if raw_events:
        return [(float(event["t_open_s"]), float(event["t_close_s"])) for event in raw_events]

    t_open = cfg.get("t_open_s", schedule.get("t_open_s"))
    t_close = cfg.get("t_close_s", schedule.get("t_close_s"))
    if t_open is not None and t_close is not None:
        return [(float(t_open), float(t_close))]

    start = float(cfg.get("start_time_s", 0.0)) + float(cfg.get("delay_s", 0.0))
    if "open_duration_s" in cfg:
        duration = float(cfg["open_duration_s"])
    elif "open_duration_s" in schedule:
        duration = float(schedule["open_duration_s"])
    else:
        duration = float(cfg.get("hold_time_s", 60.0))
    return [(start, start + max(duration, 0.0))]


def door_open_fraction(cfg: dict[str, Any], time_s: float) -> float:
    for t_open, t_close in _door_events(cfg):
        if t_open <= time_s < t_close:
            fraction = float(cfg.get("opening_fraction", cfg.get("f_open", 1.0)))
            return float(np.clip(fraction, 0.0, 1.0))
    return 0.0


def tian_geometry(cfg: dict[str, Any], opening_fraction: float) -> dict[str, float]:
    door = cfg.get("door", {})
    room = cfg.get("room", {})
    w_d = _value_from_nested(cfg, ("W_d",), ("door_width_m",), default=door.get("width_m"))
    h_d = _value_from_nested(cfg, ("H_d",), ("door_height_m",), default=door.get("height_m"))
    w_c = _value_from_nested(cfg, ("W_c",), ("room_width_m",), default=room.get("width_m"))
    l_c = _value_from_nested(cfg, ("L_c",), ("room_length_m",), ("room_depth_m",), default=room.get("length_m", room.get("depth_m")))
    h_c = _value_from_nested(cfg, ("H_c",), ("room_height_m",), default=room.get("height_m"))

    effective_width = max(w_d * opening_fraction, 1.0e-9)
    a_d = w_d * h_d * opening_fraction
    a_flow = a_d / 2.0
    half_height = h_d / 2.0
    p_wetted = 2.0 * (effective_width + half_height)
    d_e = 4.0 * max(a_flow, 1.0e-12) / max(p_wetted, 1.0e-12)

    if "effective_length_m" in cfg:
        l_el = float(cfg["effective_length_m"])
        l_el = float(np.clip(l_el, 1.0e-9, l_c))
    else:
        l_max = cfg.get("L_max_m", cfg.get("l_max_m"))
        if l_max is None:
            l_max = l_c
        l_el = min(float(l_max), l_c)
        l_el = float(np.clip(l_el, 0.5 * l_c, l_c))

    return {
        "W_d": w_d,
        "H_d": h_d,
        "W_c": w_c,
        "L_c": l_c,
        "H_c": h_c,
        "A_d": a_d,
        "A_flow": a_flow,
        "D_e": d_e,
        "L_el": l_el,
        "V_eff": max(w_d * opening_fraction * l_el * h_c, 1.0e-9),
        "V_c": max(w_c * l_c * h_c, 1.0e-9),
        "l_tl": max(2.0 * (l_el + h_c - h_d), 1.0e-9),
    }


def _stage_fraction(cumulative_volume_m3: float, v_eff_m3: float) -> float:
    return 1.0 if cumulative_volume_m3 >= v_eff_m3 else 0.0


def _state_derivative(
    cfg: dict[str, Any],
    state: TianInfiltrationState,
    geom: dict[str, float],
    rho_i: float,
    rho_o: float,
) -> tuple[float, float, float, float, float]:
    lambda_f = float(cfg.get("lambda_f", 0.025))
    xi_sum = float(cfg.get("xi_sum", 4.0))
    effectiveness = float(np.clip(cfg.get("effectiveness", cfg.get("E", 0.0)), 0.0, 1.0))

    v = max(float(state.velocity_m_s), 0.0)
    rho = max(float(state.density_kg_m3), 1.0e-9)
    stage2 = _stage_fraction(state.cumulative_volume_m3, geom["V_eff"])
    rho_exit = (1.0 - stage2) * rho_i + stage2 * rho

    delta_p_g = (geom["H_d"] / 4.0) * G * (rho - rho_o)
    delta_p_d = (geom["H_d"] / 4.0) * G * (rho_exit - rho_o)
    resistance = lambda_f * (geom["l_tl"] / max(geom["D_e"], 1.0e-9)) + xi_sum
    delta_p_f = resistance * rho * v * v / 2.0
    dv_dt = ((delta_p_g + delta_p_d - delta_p_f) * geom["A_flow"]) / max(rho * geom["V_eff"], 1.0e-9)

    q_unprotected = v * geom["A_flow"]
    q_effective = (1.0 - effectiveness) * q_unprotected
    target_rho = (1.0 - stage2) * rho_i + stage2 * rho
    drho_dt = q_effective * (rho_o - target_rho) / geom["V_c"]
    return dv_dt, drho_dt, q_effective, q_unprotected, stage2


def _integrate_open_step(
    cfg: dict[str, Any],
    state: TianInfiltrationState,
    dt_s: float,
    geom: dict[str, float],
    rho_i: float,
    rho_o: float,
) -> float:
    max_step = max(float(cfg.get("integration_max_step_s", 0.1)), 1.0e-4)
    steps = max(1, int(math.ceil(max(dt_s, 0.0) / max_step)))
    h = max(dt_s, 0.0) / steps
    q_effective = 0.0
    lo = min(rho_i, rho_o) * 0.5
    hi = max(rho_i, rho_o) * 1.5
    for _ in range(steps):
        dv_dt, drho_dt, q_effective, _, _ = _state_derivative(cfg, state, geom, rho_i, rho_o)
        state.velocity_m_s = max(state.velocity_m_s + h * dv_dt, 0.0)
        state.density_kg_m3 = float(np.clip(state.density_kg_m3 + h * drho_dt, lo, hi))
        state.cumulative_volume_m3 = max(state.cumulative_volume_m3 + h * q_effective, 0.0)
    return q_effective


def advance_tian_infiltration(
    cfg: dict[str, Any],
    state: TianInfiltrationState,
    time_s: float,
    dt_s: float,
    room_c: float,
    outdoor_c: float,
    pressure_i_pa: float,
    pressure_o_pa: float,
    omega_room: float | None = None,
    omega_outdoor: float | None = None,
) -> dict[str, float]:
    fraction = door_open_fraction(cfg, time_s)
    room_k = float(room_c) + KELVIN_OFFSET
    outdoor_k = float(outdoor_c) + KELVIN_OFFSET
    if omega_room is None:
        omega_room = humidity_ratio_from_config(cfg, "room", room_k, pressure_i_pa)
    if omega_outdoor is None:
        omega_outdoor = humidity_ratio_from_config(cfg, "outdoor", outdoor_k, pressure_o_pa)

    rho_i = air_density(room_k, pressure_i_pa, omega_room)
    rho_o = air_density(outdoor_k, pressure_o_pa, omega_outdoor)

    if fraction <= 0.0:
        state.velocity_m_s = 0.0
        state.density_kg_m3 = rho_i
        state.cumulative_volume_m3 = 0.0
        state.was_open = False
        result = zero_infiltration_result()
        result["region_density_kg_m3"] = state.density_kg_m3
        return result

    if not state.was_open:
        state.velocity_m_s = 0.0
        state.density_kg_m3 = rho_i
        state.cumulative_volume_m3 = 0.0
        state.was_open = True

    geom = tian_geometry(cfg, fraction)
    q_effective = _integrate_open_step(cfg, state, dt_s, geom, rho_i, rho_o)
    _, _, q_effective, q_unprotected, stage2 = _state_derivative(cfg, state, geom, rho_i, rho_o)

    cp_air = float(cfg.get("Cp_air", CP_AIR))
    sensible_w = q_effective * rho_o * cp_air * (outdoor_k - room_k)
    latent_w = 0.0
    if omega_outdoor is not None and omega_room is not None:
        h_latent = float(cfg.get("h_latent_freezing", H_FG + H_FUSION) if room_k < KELVIN_OFFSET else cfg.get("h_fg", H_FG))
        latent_w = q_effective * rho_o * max(omega_outdoor - omega_room, 0.0) * h_latent
    total_w = sensible_w + latent_w

    return {
        "room_w": total_w,
        "dock_w": -total_w,
        "q_m3_s": q_effective,
        "q_unprotected_m3_s": q_unprotected,
        "q_sensible_w": sensible_w,
        "q_latent_w": latent_w,
        "q_total_w": total_w,
        "cumulative_volume_m3": state.cumulative_volume_m3,
        "velocity_m_s": state.velocity_m_s,
        "region_density_kg_m3": state.density_kg_m3,
        "rho_indoor_kg_m3": rho_i,
        "rho_outdoor_kg_m3": rho_o,
        "omega_room_kg_kg_da": float(omega_room) if omega_room is not None else 0.0,
        "omega_outdoor_kg_kg_da": float(omega_outdoor) if omega_outdoor is not None else 0.0,
        "stage": 1.0 + stage2,
        "door_open_fraction": fraction,
        "effective_length_m": geom["L_el"],
    }
