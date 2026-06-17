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


@dataclass(frozen=True)
class DoorEvent:
    t_open_s: float
    t_close_s: float
    opening_fraction: float
    ramp_open_s: float = 0.0
    ramp_close_s: float = 0.0
    interval_s: float | None = None
    repeat_count: int | None = None
    repeat_until_s: float | None = None


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
        "effective_volume_m3": 0.0,
        "maximum_effective_length_m": 0.0,
    }


def apply_infiltration_load_application(cfg: dict[str, Any], load_w: float) -> tuple[float, float]:
    mode = str(cfg.get("load_application", cfg.get("application", "room_dock_exchange"))).lower()
    if mode in {"room_dock_exchange", "exchange", "room_to_dock_exchange"}:
        return load_w, -load_w
    if mode in {"dock_only", "loading_dock_only", "dock"}:
        return 0.0, load_w
    if mode in {"room_only", "refrigerated_space_only", "cold_room_only"}:
        return load_w, 0.0
    raise ValueError(f"Unsupported infiltration load_application: {mode}")


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


def _effective_length_limit(cfg: dict[str, Any], room_length_m: float) -> float:
    l_max = cfg.get("L_max_m", cfg.get("l_max_m"))
    if l_max is None:
        return room_length_m
    return float(np.clip(float(l_max), 1.0e-9, room_length_m))


def _door_spread_effective_plan_area(
    room_width_m: float,
    room_length_m: float,
    open_door_width_m: float,
    spread_angle_deg: float,
) -> float:
    width_0 = float(np.clip(open_door_width_m, 1.0e-9, room_width_m))
    length = max(float(room_length_m), 1.0e-9)
    width_room = max(float(room_width_m), width_0)
    spread = math.tan(math.radians(float(spread_angle_deg)))
    if spread <= 1.0e-9:
        return width_0 * length

    distance_to_full_width = max((width_room - width_0) / (2.0 * spread), 0.0)
    spreading_length = min(length, distance_to_full_width)
    area = width_0 * spreading_length + spread * spreading_length * spreading_length
    if length > spreading_length:
        area += width_room * (length - spreading_length)
    return max(area, 1.0e-9)


def _tian_empirical_max_effective_length(
    effective_diameter_m: float,
    flow_area_m2: float,
    nozzle_constant: float,
) -> float:
    diameter = max(float(effective_diameter_m), 1.0e-9)
    area = max(float(flow_area_m2), 1.0e-12)
    area_ratio = math.sqrt(area) / diameter
    exponent = 0.147 * area_ratio - 0.133
    base = 3.58 * area_ratio + float(nozzle_constant)
    return diameter * max(base, 1.0e-12) ** exponent


def _tian_effective_length_and_volume(
    cfg: dict[str, Any],
    room_width_m: float,
    room_length_m: float,
    room_height_m: float,
    open_door_width_m: float,
    effective_diameter_m: float,
    flow_area_m2: float,
) -> tuple[float, float, float]:
    if "effective_length_m" in cfg:
        length_m = float(np.clip(float(cfg["effective_length_m"]), 1.0e-9, room_length_m))
        volume_m3 = open_door_width_m * length_m * room_height_m
        return length_m, max(volume_m3, 1.0e-9), length_m

    length_limit_m = _effective_length_limit(cfg, room_length_m)
    model = str(cfg.get("effective_length_model", "tian_empirical")).lower()
    if model in {"room_depth", "legacy", "full_depth"}:
        length_m = length_limit_m
        volume_m3 = open_door_width_m * length_m * room_height_m
        return length_m, max(volume_m3, 1.0e-9), length_m

    if model in {"tian_empirical", "tian", "paper", "jet_empirical"}:
        nozzle_constant = float(cfg.get("effective_nozzle_constant", cfg.get("a_effective_length", 0.1)))
        l_max = _tian_empirical_max_effective_length(effective_diameter_m, flow_area_m2, nozzle_constant)
        length_m = min(l_max, length_limit_m)
        volume_m3 = open_door_width_m * length_m * room_height_m
        return float(np.clip(length_m, 1.0e-9, room_length_m)), max(volume_m3, 1.0e-9), l_max

    if model in {"door_spread", "spread", "wedge", "tian_spread"}:
        angle_deg = float(cfg.get("effective_spread_angle_deg", cfg.get("spread_angle_deg", 30.0)))
        plan_area_m2 = _door_spread_effective_plan_area(
            room_width_m=room_width_m,
            room_length_m=length_limit_m,
            open_door_width_m=open_door_width_m,
            spread_angle_deg=angle_deg,
        )
        length_m = plan_area_m2 / max(room_width_m, 1.0e-9)
        volume_m3 = plan_area_m2 * room_height_m
        length_m = float(np.clip(length_m, 1.0e-9, room_length_m))
        return length_m, max(volume_m3, 1.0e-9), length_m

    raise ValueError(f"Unsupported Tian effective_length_model: {model}")


def _first_seconds(
    sources: tuple[dict[str, Any], ...],
    second_keys: tuple[str, ...],
    hour_keys: tuple[str, ...] = (),
    default: float | None = None,
) -> float | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in second_keys:
            if key in source:
                return float(source[key])
        for key in hour_keys:
            if key in source:
                return 3600.0 * float(source[key])
    return default


def _first_float(sources: tuple[dict[str, Any], ...], keys: tuple[str, ...], default: float) -> float:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                return float(source[key])
    return default


def _first_int(sources: tuple[dict[str, Any], ...], keys: tuple[str, ...]) -> int | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                return int(source[key])
    return None


def _door_event_from_config(event: dict[str, Any], schedule: dict[str, Any], cfg: dict[str, Any]) -> DoorEvent:
    sources = (event, schedule, cfg)

    t_open = _first_seconds(
        sources,
        second_keys=("t_open_s",),
        hour_keys=("t_open_h",),
    )
    if t_open is None:
        t_open = _first_seconds(
            sources,
            second_keys=("start_time_s",),
            hour_keys=("start_time_h",),
            default=0.0,
        )
        delay_s = _first_seconds(
            sources,
            second_keys=("delay_s",),
            hour_keys=("delay_h",),
            default=0.0,
        )
        t_open = float(t_open or 0.0) + float(delay_s or 0.0)

    duration_s = _first_seconds(
        sources,
        second_keys=("open_duration_s", "duration_s", "hold_time_s"),
        hour_keys=("open_duration_h", "duration_h", "hold_time_h"),
    )
    t_close = _first_seconds(
        sources,
        second_keys=("t_close_s", "end_time_s"),
        hour_keys=("t_close_h", "end_time_h"),
    )
    if t_close is None:
        if duration_s is None:
            duration_s = 60.0
        t_close = t_open + max(float(duration_s), 0.0)

    peak_fraction = _first_float(sources, ("opening_fraction", "f_open"), default=1.0)
    ramp_s = _first_seconds(
        sources,
        second_keys=("door_ramp_time_s", "ramp_time_s"),
        hour_keys=("door_ramp_time_h", "ramp_time_h"),
        default=0.0,
    )
    ramp_open_s = _first_seconds(
        sources,
        second_keys=("door_ramp_open_time_s", "ramp_open_time_s"),
        hour_keys=("door_ramp_open_time_h", "ramp_open_time_h"),
        default=ramp_s,
    )
    ramp_close_s = _first_seconds(
        sources,
        second_keys=("door_ramp_close_time_s", "ramp_close_time_s"),
        hour_keys=("door_ramp_close_time_h", "ramp_close_time_h"),
        default=ramp_s,
    )
    repeat_count = _first_int(sources, ("repeat_count", "count", "n_events"))
    repeat_until_s = _first_seconds(
        sources,
        second_keys=("repeat_until_s", "schedule_end_s"),
        hour_keys=("repeat_until_h", "schedule_end_h"),
    )
    repeat_for_s = _first_seconds(
        sources,
        second_keys=("repeat_for_s", "schedule_duration_s"),
        hour_keys=("repeat_for_h", "schedule_duration_h"),
    )
    if repeat_until_s is None and repeat_for_s is not None:
        repeat_until_s = float(t_open) + max(float(repeat_for_s), 0.0)
    interval_s = _first_seconds(
        sources,
        second_keys=("interval_s", "period_s"),
        hour_keys=("interval_h", "period_h"),
    )
    if interval_s is None and repeat_count is not None and repeat_count > 1 and repeat_for_s is not None:
        interval_s = max(float(repeat_for_s) / repeat_count, 0.0)

    return DoorEvent(
        t_open_s=float(t_open),
        t_close_s=float(t_close),
        opening_fraction=float(np.clip(peak_fraction, 0.0, 1.0)),
        ramp_open_s=max(float(ramp_open_s or 0.0), 0.0),
        ramp_close_s=max(float(ramp_close_s or 0.0), 0.0),
        interval_s=max(float(interval_s), 0.0) if interval_s is not None else None,
        repeat_count=max(int(repeat_count), 0) if repeat_count is not None else None,
        repeat_until_s=float(repeat_until_s) if repeat_until_s is not None else None,
    )


def _door_events(cfg: dict[str, Any]) -> list[DoorEvent]:
    raw_schedule = cfg.get("schedule", {})
    schedule = raw_schedule if isinstance(raw_schedule, dict) else {}
    raw_events = schedule.get("events", cfg.get("events"))
    if raw_events:
        return [_door_event_from_config(event, schedule, cfg) for event in raw_events]

    return [_door_event_from_config({}, schedule, cfg)]


def _single_opening_fraction(event: DoorEvent, time_s: float, t_open_s: float) -> float:
    t_close_s = t_open_s + max(event.t_close_s - event.t_open_s, 0.0)
    if not (t_open_s <= time_s < t_close_s):
        return 0.0

    duration_s = max(t_close_s - t_open_s, 0.0)
    if duration_s <= 0.0:
        return 0.0

    ramp_open_s = min(event.ramp_open_s, duration_s)
    ramp_close_s = min(event.ramp_close_s, max(duration_s - ramp_open_s, 0.0))

    if ramp_open_s > 0.0 and time_s < t_open_s + ramp_open_s:
        return event.opening_fraction * (time_s - t_open_s) / ramp_open_s

    if ramp_close_s > 0.0 and time_s >= t_close_s - ramp_close_s:
        return event.opening_fraction * (t_close_s - time_s) / ramp_close_s

    return event.opening_fraction


def _event_start_is_repeated(event: DoorEvent, index: int, t_open_s: float) -> bool:
    if index < 0:
        return False
    if event.repeat_count is not None and index >= event.repeat_count:
        return False
    return event.repeat_until_s is None or t_open_s < event.repeat_until_s


def _ramped_fraction(event: DoorEvent, time_s: float) -> float:
    if event.interval_s is None or event.interval_s <= 0.0:
        return _single_opening_fraction(event, time_s, event.t_open_s)
    if time_s < event.t_open_s:
        return 0.0

    if event.interval_s <= 0.0:
        return 0.0

    index = int(math.floor((time_s - event.t_open_s) / event.interval_s))
    fractions = [
        _single_opening_fraction(event, time_s, event.t_open_s + candidate * event.interval_s)
        for candidate in (index - 1, index)
        if _event_start_is_repeated(event, candidate, event.t_open_s + candidate * event.interval_s)
    ]
    return max(fractions, default=0.0)


def door_open_fraction(cfg: dict[str, Any], time_s: float) -> float:
    fraction = max(_ramped_fraction(event, time_s) for event in _door_events(cfg))
    return float(np.clip(fraction, 0.0, 1.0))


def _event_start_allowed(event: DoorEvent, index: int, t_open_s: float) -> bool:
    if event.interval_s is None or event.interval_s <= 0.0:
        return index == 0
    return _event_start_is_repeated(event, index, t_open_s)


def scheduled_door_event_starts_between(cfg: dict[str, Any], start_s: float, end_s: float) -> list[float]:
    start = float(start_s)
    end = float(end_s)
    if end < start:
        start, end = end, start

    starts: list[float] = []
    for event in _door_events(cfg):
        if event.interval_s is None or event.interval_s <= 0.0:
            if start <= event.t_open_s <= end:
                starts.append(event.t_open_s)
            continue

        interval_s = max(float(event.interval_s), 1.0e-9)
        first_index = max(0, int(math.ceil((start - event.t_open_s) / interval_s)))
        last_index = int(math.floor((end - event.t_open_s) / interval_s))
        for index in range(first_index, last_index + 1):
            t_open = event.t_open_s + index * interval_s
            if start <= t_open <= end and _event_start_allowed(event, index, t_open):
                starts.append(t_open)

    return sorted(starts)


def scheduled_precool_fraction(cfg: dict[str, Any], time_s: float, lead_time_s: float) -> float:
    lead_s = max(float(lead_time_s), 0.0)
    if lead_s <= 0.0:
        return 0.0

    time = float(time_s)
    if door_open_fraction(cfg, time) > 0.0:
        return 0.0
    for t_open in scheduled_door_event_starts_between(cfg, time, time + lead_s):
        if t_open - lead_s <= time < t_open:
            return 1.0
    return 0.0


def tian_geometry(cfg: dict[str, Any], opening_fraction: float) -> dict[str, float]:
    door = cfg.get("door", {})
    indoor_source = str(cfg.get("indoor_source", "room")).lower()
    fallback_room = cfg.get("room", {})
    if "indoor" in cfg:
        room = cfg["indoor"]
    elif indoor_source in {"dock", "loading_dock"} and "dock" in cfg:
        room = cfg["dock"]
    else:
        room = fallback_room
    w_d = _value_from_nested(cfg, ("W_d",), ("door_width_m",), default=door.get("width_m"))
    h_d = _value_from_nested(cfg, ("H_d",), ("door_height_m",), default=door.get("height_m"))
    w_c = _value_from_nested(
        cfg,
        ("W_c",),
        ("room_width_m",),
        default=room.get("width_m", fallback_room.get("width_m")),
    )
    l_c = _value_from_nested(
        cfg,
        ("L_c",),
        ("room_length_m",),
        ("room_depth_m",),
        default=room.get("length_m", room.get("depth_m", fallback_room.get("length_m", fallback_room.get("depth_m")))),
    )
    h_c = _value_from_nested(
        cfg,
        ("H_c",),
        ("room_height_m",),
        default=room.get("height_m", fallback_room.get("height_m")),
    )

    effective_width = max(w_d * opening_fraction, 1.0e-9)
    a_d = w_d * h_d * opening_fraction
    a_flow = a_d / 2.0
    half_height = h_d / 2.0
    p_wetted = 2.0 * (effective_width + half_height)
    d_e = 4.0 * max(a_flow, 1.0e-12) / max(p_wetted, 1.0e-12)

    l_el, v_eff, l_max = _tian_effective_length_and_volume(
        cfg,
        room_width_m=w_c,
        room_length_m=l_c,
        room_height_m=h_c,
        open_door_width_m=effective_width,
        effective_diameter_m=d_e,
        flow_area_m2=a_flow,
    )

    return {
        "W_d": w_d,
        "H_d": h_d,
        "W_c": w_c,
        "L_c": l_c,
        "H_c": h_c,
        "A_d": a_d,
        "A_flow": a_flow,
        "D_e": d_e,
        "L_max": l_max,
        "L_el": l_el,
        "V_eff": v_eff,
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
        omega_room = humidity_ratio_from_config(cfg, "indoor", room_k, pressure_i_pa)
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
    room_w, dock_w = apply_infiltration_load_application(cfg, total_w)

    return {
        "room_w": room_w,
        "dock_w": dock_w,
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
        "effective_volume_m3": geom["V_eff"],
        "maximum_effective_length_m": geom["L_max"],
    }
