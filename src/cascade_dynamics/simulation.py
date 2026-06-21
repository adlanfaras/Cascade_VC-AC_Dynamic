from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
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
from .compressor_map import (
    AIR_PERFORMANCE_MAP_VALIDITY,
    air_performance_map_model,
    ammonia_compressor_eta_is,
    ammonia_compressor_map,
    compressor_uses_ammonia_eta_is_map,
    screw_compressor_pressure_ratio_map,
    volumetric_flow_from_head,
)
from .components import positive_lmtd, turbine_actual_enthalpy
from .fluids import configure_property_backend_from_config, h_refrigerant_liquid, p_sat, props_si
from .humid_air import humid_air_state, saturated_room_humidity_ratio, state_at_enthalpy, state_at_entropy
from .model import (
    CP_DOCK_AIR,
    CascadeSystemModel,
    compressor_discharge_pressure_from_power,
    compressor_eta_is_from_map_power,
    compressor_mass_flow_positive_displacement,
    compressor_uses_map_power_pressure_lift,
    compressor_volumetric_efficiency_clearance,
    expansion_valve_flow_coefficient,
    expansion_valve_flow_factor,
)
from .numerics import NewtonSolveError, newton_raphson_fd


KELVIN_OFFSET = 273.15
STARTUP_CACHE_VERSION = 28

STATE_INDEX = {
    "room_c": 0,
    "sink_c": 1,
    "t3_c": 2,
    "t4_c": 3,
    "t6_c": 4,
    "tevap_c": 5,
    "tcond_c": 6,
    "m_ref_cascade_kg_s": 7,
    "m_ref_dock_kg_s": 8,
    "dock_c": 9,
    "receiver_mass_kg": 10,
}

SUPPORTED_SYSTEM_MODES = {"cascade", "air_cycle", "air", "vcc", "vapor_compression"}
SCREW_PRESSURE_RATIO_COMPRESSOR_MODELS = {
    "screw_pressure_ratio_polynomial",
    "screw_polynomial",
    "pressure_ratio_polynomial_screw",
}

DEFAULT_STARTUP_FREE_PARAMETERS = [
    {"path": "vcc_cycle.condenser_ua_w_k", "min": 1000.0, "max": 100000.0},
    {"path": "vcc_cycle.cascade_ua_w_k", "min": 100.0, "max": 100000.0},
    {"path": "boundary_conditions.load_before_w", "min": 1000.0, "max": 200000.0},
    {"path": "vcc_cycle.expansion_valves.cascade.opening", "min": 0.05, "max": 1.0, "freeze_in_transient": False},
    {"path": "vcc_cycle.expansion_valves.dock.opening", "min": 0.05, "max": 1.0, "freeze_in_transient": False},
    {"path": "air_cycle.pressure_ratio", "min": 1.01, "max": 1.6},
    {"path": "air_cycle.compressor_mass_flow.speed_rpm", "min": 10000.0, "max": 18000.0, "freeze_in_transient": False},
    {"path": "vcc_cycle.compressor.speed_rpm", "min": 1226.0, "max": 1610.0, "freeze_in_transient": False},
]
DEFAULT_STARTUP_FREE_STATE_VARIABLES = ["t3_c", "t4_c", "t6_c", "m_ref_kg_s"]
PAPER_DESIGN_SOLVED_PATHS = [
    "air_cycle.pressure_ratio",
    "air_cycle.compressor_mass_flow.speed_rpm",
    "air_cycle.compressor_mass_flow.fixed_m_dot_kg_s",
    "air_cycle.compressor_mass_flow.damper_opening",
    "air_cycle.compressor_mass_flow.damper_resistance_head_coefficient",
    "air_cycle.compressor_mass_flow.system_static_head_m",
    "air_cycle.regenerator_ua_w_k",
    "vcc_cycle.cascade_ua_w_k",
    "vcc_cycle.dock_evaporator.ua_w_k",
    "vcc_cycle.condenser_ua_w_k",
    "vcc_cycle.compressor.speed_rpm",
    "vcc_cycle.compressor.eta_is",
    "vcc_cycle.compressor.displacement_m3_per_rev",
    "vcc_cycle.expansion_valves.cascade.opening",
    "vcc_cycle.expansion_valves.cascade.flow_coefficient_m2",
    "vcc_cycle.expansion_valves.dock.opening",
    "vcc_cycle.expansion_valves.dock.flow_coefficient_m2",
    "vcc_cycle.receiver.volume_m3",
    "vcc_cycle.receiver.initial_mass_kg",
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


def _receiver_config(config: dict[str, Any]) -> dict[str, Any]:
    vcc_cfg = config.get("vcc_cycle", {})
    receiver_cfg = vcc_cfg.get("receiver")
    if isinstance(receiver_cfg, dict):
        return receiver_cfg
    receiver_cfg = vcc_cfg.get("high_pressure_receiver")
    if isinstance(receiver_cfg, dict):
        return receiver_cfg
    return {}


def _vcc_layout(config: dict[str, Any]) -> str:
    vcc_cfg = config.get("vcc_cycle", {})
    layout = vcc_cfg.get("layout", vcc_cfg.get("configuration", vcc_cfg.get("cycle_layout", "")))
    return str(layout or "").strip().lower()


def _receiver_enabled(config: dict[str, Any]) -> bool:
    return bool(_receiver_config(config).get("enabled", False)) or _vcc_layout(config) in {"hpr", "high_pressure_receiver"}


def _low_pressure_receiver_config(config: dict[str, Any]) -> dict[str, Any]:
    vcc_cfg = config.get("vcc_cycle", {})
    receiver_cfg = vcc_cfg.get("low_pressure_receiver", vcc_cfg.get("lpr", {}))
    return receiver_cfg if isinstance(receiver_cfg, dict) else {}


def _low_pressure_receiver_enabled(config: dict[str, Any]) -> bool:
    return bool(_low_pressure_receiver_config(config).get("enabled", False)) or _vcc_layout(config) in {
        "lpr",
        "low_pressure_receiver",
    }


def _vcc_evaporators_are_series(config: dict[str, Any]) -> bool:
    vcc_cfg = config.get("vcc_cycle", {})
    lpr_cfg = _low_pressure_receiver_config(config)
    arrangement = vcc_cfg.get(
        "evaporator_arrangement",
        vcc_cfg.get(
            "evaporator_topology",
            lpr_cfg.get("evaporator_arrangement", lpr_cfg.get("evaporator_topology", "")),
        ),
    )
    arrangement_key = str(arrangement or "").strip().lower().replace("-", "_").replace(" ", "_")
    return arrangement_key in {
        "series",
        "series_dock_cascade",
        "dock_cascade_series",
        "dock_then_cascade",
        "exp_dock_cascade",
        "expansion_dock_cascade",
    }


def _effective_refrigerant_mass_flow(config: dict[str, Any], m_ref_cascade: float, m_ref_dock: float) -> float:
    if _vcc_evaporators_are_series(config):
        return 0.5 * (max(float(m_ref_cascade), 0.0) + max(float(m_ref_dock), 0.0))
    return max(float(m_ref_cascade), 0.0) + max(float(m_ref_dock), 0.0)


def _compressor_uses_saturated_suction(config: dict[str, Any]) -> bool:
    return _low_pressure_receiver_enabled(config) and bool(
        _low_pressure_receiver_config(config).get("force_saturated_suction", False)
    )


def _lpr_subcooling_control_enabled(config: dict[str, Any]) -> bool:
    receiver_cfg = _low_pressure_receiver_config(config)
    return _low_pressure_receiver_enabled(config) and bool(
        receiver_cfg.get("control_subcooling_with_eev", receiver_cfg.get("dynamic_subcooling", False))
    )


def _lpr_inventory_enabled(config: dict[str, Any]) -> bool:
    receiver_cfg = _low_pressure_receiver_config(config)
    model = str(receiver_cfg.get("model", receiver_cfg.get("inventory_model", ""))).strip().lower().replace("-", "_").replace(" ", "_")
    return _low_pressure_receiver_enabled(config) and (
        model in {"dynamic_inventory", "liquid_inventory", "two_phase_inventory"}
        or bool(receiver_cfg.get("liquid_inventory_enabled", receiver_cfg.get("dynamic_inventory", False)))
    )


def _lpr_initial_subcooling_k(config: dict[str, Any]) -> float:
    receiver_cfg = _low_pressure_receiver_config(config)
    guess = config.get("initial_guess", {})
    return max(
        float(
            guess.get(
                "subcooling_k",
                receiver_cfg.get(
                    "initial_subcooling_k",
                    receiver_cfg.get(
                        "subcooling_setpoint_k",
                        receiver_cfg.get("target_subcooling_k", config.get("vcc_cycle", {}).get("subcooling_k", 0.0)),
                    ),
                ),
            )
        ),
        0.0,
    )


def _lpr_initial_masses_kg(config: dict[str, Any]) -> tuple[float, float]:
    receiver_cfg = _low_pressure_receiver_config(config)
    guess = config.get("initial_guess", {})
    liquid_guess = guess.get(
        "lpr_liquid_mass_kg",
        receiver_cfg.get("initial_liquid_mass_kg", receiver_cfg.get("liquid_mass_kg")),
    )
    vapor_guess = guess.get(
        "lpr_vapor_mass_kg",
        receiver_cfg.get("initial_vapor_mass_kg", receiver_cfg.get("vapor_mass_kg")),
    )
    if liquid_guess is not None and vapor_guess is not None:
        return max(float(liquid_guess), 0.0), max(float(vapor_guess), 0.0)

    ref_fluid = config.get("fluids", {}).get("refrigerant", "Ammonia")
    startup_targets = config.get("simulation", {}).get("startup_initialization", {}).get("targets", {})
    tevap_c = float(startup_targets.get("tevap_c", guess.get("tevap_c", -10.0)))
    tevap_k = tevap_c + KELVIN_OFFSET
    rho_l = props_si("D", "T", tevap_k, "Q", 0.0, ref_fluid)
    rho_v = props_si("D", "T", tevap_k, "Q", 1.0, ref_fluid)
    volume = max(float(receiver_cfg.get("volume_m3", receiver_cfg.get("internal_volume_m3", 0.0))), 0.0)
    initial_fill = float(receiver_cfg.get("initial_liquid_fill_fraction", 0.10))
    initial_fill = float(np.clip(initial_fill, 0.0, 1.0))
    liquid_mass = max(float(liquid_guess), 0.0) if liquid_guess is not None else initial_fill * volume * max(rho_l, 0.0)
    vapor_mass = max(float(vapor_guess), 0.0) if vapor_guess is not None else (1.0 - initial_fill) * volume * max(rho_v, 0.0)
    return liquid_mass, vapor_mass


def _receiver_initial_mass_kg(config: dict[str, Any]) -> float:
    receiver_cfg = _receiver_config(config)
    guess = config.get("initial_guess", {})
    return max(
        float(
            guess.get(
                "receiver_mass_kg",
                receiver_cfg.get("initial_mass_kg", receiver_cfg.get("mass_kg", 0.0)),
            )
        ),
        0.0,
    )


def _configure_receiver_defaults(config: dict[str, Any]) -> None:
    if not _receiver_enabled(config):
        return

    receiver_cfg = _receiver_config(config)
    guess = config.setdefault("initial_guess", {})
    ref_fluid = config.get("fluids", {}).get("refrigerant", "Ammonia")
    startup_targets = config.get("simulation", {}).get("startup_initialization", {}).get("targets", {})
    tcond_c = float(startup_targets.get("tcond_c", guess.get("tcond_c", 40.0)))
    tcond_k = tcond_c + KELVIN_OFFSET
    rho_l = props_si("D", "T", tcond_k, "Q", 0.0, ref_fluid)
    rho_v = props_si("D", "T", tcond_k, "Q", 1.0, ref_fluid)
    initial_fill = float(receiver_cfg.get("initial_liquid_fill_fraction", 0.5))
    initial_fill = float(np.clip(initial_fill, 0.0, 1.0))
    mixture_density = rho_v + initial_fill * (rho_l - rho_v)

    has_mass = "receiver_mass_kg" in guess or "initial_mass_kg" in receiver_cfg or "mass_kg" in receiver_cfg
    has_volume = "volume_m3" in receiver_cfg or "internal_volume_m3" in receiver_cfg
    if has_mass:
        initial_mass = _receiver_initial_mass_kg(config)
    elif has_volume:
        volume = max(float(receiver_cfg.get("volume_m3", receiver_cfg.get("internal_volume_m3", 0.0))), 0.0)
        initial_mass = volume * max(mixture_density, 1.0e-9)
    else:
        residence_s = float(receiver_cfg.get("initial_residence_time_s", receiver_cfg.get("residence_time_s", 5.0)))
        initial_mass = max(float(guess.get("m_ref_kg_s", 0.1)), 1.0e-6) * max(residence_s, 0.0)

    initial_mass = max(float(initial_mass), 0.0)
    receiver_cfg["initial_mass_kg"] = initial_mass
    guess["receiver_mass_kg"] = initial_mass

    if not has_volume:
        volume = initial_mass / max(mixture_density, 1.0e-9)
        receiver_cfg["volume_m3"] = max(volume, 1.0e-9)
        receiver_cfg["auto_sized_volume_m3"] = receiver_cfg["volume_m3"]


def _configure_lpr_subcooling_defaults(config: dict[str, Any]) -> None:
    if not _lpr_subcooling_control_enabled(config):
        return
    receiver_cfg = _low_pressure_receiver_config(config)
    guess = config.setdefault("initial_guess", {})
    target = max(
        float(
            receiver_cfg.get(
                "subcooling_setpoint_k",
                receiver_cfg.get("target_subcooling_k", config.get("vcc_cycle", {}).get("subcooling_k", 0.0)),
            )
        ),
        0.0,
    )
    receiver_cfg.setdefault("initial_subcooling_k", target)
    guess.setdefault("subcooling_k", receiver_cfg["initial_subcooling_k"])


def _configure_lpr_inventory_defaults(config: dict[str, Any]) -> None:
    if not _lpr_inventory_enabled(config):
        return

    receiver_cfg = _low_pressure_receiver_config(config)
    guess = config.setdefault("initial_guess", {})
    ref_fluid = config.get("fluids", {}).get("refrigerant", "Ammonia")
    startup_targets = config.get("simulation", {}).get("startup_initialization", {}).get("targets", {})
    tevap_c = float(startup_targets.get("tevap_c", guess.get("tevap_c", -10.0)))
    tevap_k = tevap_c + KELVIN_OFFSET
    rho_l = props_si("D", "T", tevap_k, "Q", 0.0, ref_fluid)
    rho_v = props_si("D", "T", tevap_k, "Q", 1.0, ref_fluid)
    volume = max(float(receiver_cfg.get("volume_m3", receiver_cfg.get("internal_volume_m3", 0.0))), 0.0)
    if volume <= 0.0:
        m_ref = max(float(guess.get("m_ref_kg_s", 0.1)), 1.0e-6)
        residence_s = max(float(receiver_cfg.get("initial_residence_time_s", receiver_cfg.get("residence_time_s", 5.0))), 0.0)
        fill = float(np.clip(float(receiver_cfg.get("initial_liquid_fill_fraction", 0.10)), 0.0, 0.95))
        volume = max(m_ref * residence_s / max(rho_v + fill * (rho_l - rho_v), 1.0e-9), 1.0e-9)
        receiver_cfg["volume_m3"] = volume
        receiver_cfg["auto_sized_volume_m3"] = volume

    liquid_mass, vapor_mass = _lpr_initial_masses_kg(config)
    receiver_cfg["initial_liquid_mass_kg"] = liquid_mass
    receiver_cfg["initial_vapor_mass_kg"] = vapor_mass
    guess["lpr_liquid_mass_kg"] = liquid_mass
    guess["lpr_vapor_mass_kg"] = vapor_mass

    liquid_max = max(float(receiver_cfg.get("liquid_mass_max_kg", 0.0)), 1.05 * volume * max(rho_l, 0.0), liquid_mass)
    vapor_max = max(float(receiver_cfg.get("vapor_mass_max_kg", 0.0)), 20.0 * volume * max(rho_v, 0.0), 20.0 * vapor_mass, 0.05)
    receiver_cfg.setdefault("liquid_mass_min_kg", 0.0)
    receiver_cfg["liquid_mass_max_kg"] = liquid_max
    receiver_cfg.setdefault("vapor_mass_min_kg", 0.0)
    receiver_cfg["vapor_mass_max_kg"] = vapor_max
    receiver_cfg.setdefault("liquid_mass_residual_scale_kg", max(0.01 * liquid_max, 1.0e-4))
    receiver_cfg.setdefault("vapor_mass_residual_scale_kg", max(0.05 * max(vapor_mass, volume * max(rho_v, 0.0)), 1.0e-5))
    receiver_cfg.setdefault("volume_residual_scale_m3", max(volume, 1.0e-6))


def _path_exists(data: dict[str, Any], path: str) -> bool:
    try:
        get_path(data, path)
    except KeyError:
        return False
    return True


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
    m_ref_total = float(guess["m_ref_kg_s"])
    m_ref_cascade, m_ref_dock = _split_refrigerant_mass_flow(config, m_ref_total)
    values = [
        guess["room_c"],
        guess["sink_c"],
        guess["t3_c"],
        guess["t4_c"],
        guess["t6_c"],
        guess["tevap_c"],
        guess["tcond_c"],
        m_ref_cascade,
        m_ref_dock,
        guess.get("dock_c", config["boundary_conditions"]["dock_initial_c"]),
    ]
    if _receiver_enabled(config):
        values.append(_receiver_initial_mass_kg(config))
    if _lpr_subcooling_control_enabled(config):
        values.append(_lpr_initial_subcooling_k(config))
    if _lpr_inventory_enabled(config):
        values.extend(_lpr_initial_masses_kg(config))
    return np.array(values, dtype=float)


def system_mode(config: dict[str, Any]) -> str:
    raw_mode = config.get("system_mode", config.get("mode"))
    system_cfg = config.get("system", {})
    if isinstance(system_cfg, dict):
        raw_mode = system_cfg.get("mode", raw_mode)
    mode = str(raw_mode or "cascade").strip().lower()
    if mode not in SUPPORTED_SYSTEM_MODES:
        raise ValueError(f"Unsupported system mode '{mode}'. Choose one of: {', '.join(sorted(SUPPORTED_SYSTEM_MODES))}.")
    if mode == "air":
        return "air_cycle"
    if mode == "vapor_compression":
        return "vcc"
    return mode


LUMPED_MATRIX_REGENERATOR_MODELS = {
    "lumped_matrix",
    "matrix_lumped",
    "transient_lumped_matrix",
    "lumped_matrix_transient",
}
TWO_LUMP_MATRIX_REGENERATOR_MODELS = {
    "two_lump_matrix",
    "two_lump",
    "two_node_matrix",
    "two_node_lumped_matrix",
}
TRANSIENT_REGENERATOR_MODELS = {
    "transient_distributed",
    "transient_lumped",
    "distributed_transient",
    "yang_transient",
    *LUMPED_MATRIX_REGENERATOR_MODELS,
    *TWO_LUMP_MATRIX_REGENERATOR_MODELS,
}
LUMPED_CASCADE_EXCHANGER_MODELS = {
    "lumped_capacitance",
    "lumped_capacity",
    "lumped",
    "one_cell_lumped",
    "one_cell_lumped_capacitance",
    "transient_lumped",
}


def _air_cycle_transient_regenerator_config(config: dict[str, Any]) -> dict[str, Any] | None:
    reg_cfg = config.get("air_cycle", {}).get("regenerator", {})
    if not isinstance(reg_cfg, dict):
        return None
    model = str(reg_cfg.get("model", "")).strip().lower()
    if model not in TRANSIENT_REGENERATOR_MODELS:
        return None
    return reg_cfg


def _air_cycle_transient_regenerator_cell_count(config: dict[str, Any]) -> int:
    reg_cfg = _air_cycle_transient_regenerator_config(config)
    if reg_cfg is None:
        return 0
    model = str(reg_cfg.get("model", "")).strip().lower()
    if model in LUMPED_MATRIX_REGENERATOR_MODELS:
        return max(1, int(reg_cfg.get("matrix_count", reg_cfg.get("cell_count", 1))))
    if model in TWO_LUMP_MATRIX_REGENERATOR_MODELS:
        return max(2, int(reg_cfg.get("matrix_count", reg_cfg.get("cell_count", 2))))
    return max(1, int(reg_cfg.get("cell_count", reg_cfg.get("cells", 12))))


def _air_cycle_regenerator_initial_cells_c(config: dict[str, Any], guess: dict[str, Any]) -> list[float]:
    reg_cfg = _air_cycle_transient_regenerator_config(config)
    if reg_cfg is None:
        return []

    cell_count = _air_cycle_transient_regenerator_cell_count(config)
    profile_k = reg_cfg.get("initial_solid_profile_k")
    if isinstance(profile_k, list) and profile_k:
        values = [float(value) - KELVIN_OFFSET for value in profile_k]
        if len(values) >= cell_count:
            return values[:cell_count]
        return values + [values[-1]] * (cell_count - len(values))

    profile_c = reg_cfg.get("initial_solid_profile_c")
    if isinstance(profile_c, list) and profile_c:
        values = [float(value) for value in profile_c]
        if len(values) >= cell_count:
            return values[:cell_count]
        return values + [values[-1]] * (cell_count - len(values))

    if "initial_solid_k" in reg_cfg:
        initial_c = float(reg_cfg["initial_solid_k"]) - KELVIN_OFFSET
    elif "initial_solid_c" in reg_cfg:
        initial_c = float(reg_cfg["initial_solid_c"])
    else:
        initial_c = float(guess.get("room_c", config.get("boundary_conditions", {}).get("ambient_c", 30.0)))
    return [initial_c] * cell_count


def _configure_cascade_lumped_heat_exchangers(config: dict[str, Any]) -> None:
    if system_mode(config) != "cascade":
        return

    air_cfg = config.setdefault("air_cycle", {})
    reg_cfg = air_cfg.setdefault("regenerator", {})
    if isinstance(reg_cfg, dict) and reg_cfg.get("enabled", True):
        explicit_gas_solid_ua_factor = "gas_solid_ua_factor" in reg_cfg
        reg_cfg.setdefault("model", "transient_distributed")
        reg_cfg.setdefault("cell_count", 6)
        reg_cfg.setdefault("ua_w_k", air_cfg.get("regenerator_ua_w_k", 0.0))
        reg_cfg.setdefault("cp_air_j_kg_k", 1005.0)
        reg_cfg.setdefault("gas_solid_ua_factor", 2.0)
        reg_cfg.setdefault("auto_calibrate_gas_solid_ua_factor", not explicit_gas_solid_ua_factor)
        reg_cfg.setdefault("calibrated_gas_effectiveness_max", 0.98)
        reg_cfg.setdefault("steady_effectiveness_correction", True)
        reg_cfg.setdefault(
            "steady_effectiveness_target",
            config.get("simulation", {}).get("startup_initialization", {}).get("regenerator_effectiveness", 0.9),
        )
        reg_cfg.setdefault("steady_profile_relaxation_time_s", reg_cfg.get("auto_capacitance_time_constant_s", 4.5))
        reg_cfg.setdefault("outlet_residual_scale_k", 1.0)
        reg_cfg.setdefault("solid_lower_c", -200.0)
        reg_cfg.setdefault("solid_upper_c", 150.0)
        reg_cfg.setdefault("solid_step_scale_k", 10.0)
        reg_cfg.setdefault("solid_residual_scale_k", 1.0)
        reg_cfg.setdefault("auto_capacitance", True)
        reg_cfg.setdefault("auto_capacitance_time_constant_s", 4.5)
        reg_cfg.setdefault("fluid_residence_time_s", 1.0)

    vcc_cfg = config.setdefault("vcc_cycle", {})
    cascade_cfg = vcc_cfg.setdefault("cascade_exchanger", {})
    if isinstance(cascade_cfg, dict) and cascade_cfg.get("enabled", True):
        cascade_cfg.setdefault("model", "lumped_capacitance")
        cascade_cfg.setdefault("cell_count", 1)
        cascade_cfg.setdefault("ua_w_k", vcc_cfg.get("cascade_ua_w_k", 0.0))
        cascade_cfg.setdefault("cp_air_j_kg_k", 1005.0)
        cascade_cfg.setdefault("air_side_ua_factor", 2.0)
        cascade_cfg.setdefault("refrigerant_side_ua_factor", 2.0)
        cascade_cfg.setdefault("solid_lower_c", -100.0)
        cascade_cfg.setdefault("solid_upper_c", 150.0)
        cascade_cfg.setdefault("solid_step_scale_k", 5.0)
        cascade_cfg.setdefault("solid_residual_scale_k", 1.0)
        cascade_cfg.setdefault("auto_capacitance", True)
        cascade_cfg.setdefault("auto_capacitance_time_constant_s", 5.0)
        cascade_cfg.setdefault("air_residence_time_s", 1.0)
        cascade_cfg.setdefault("refrigerant_residence_time_s", 3.0)


def _has_explicit_heat_capacity(cfg: dict[str, Any]) -> bool:
    return any(
        key in cfg
        for key in (
            "solid_capacitance_j_k",
            "capacitance_j_k",
            "solid_capacitance_profile_j_k",
            "matrix_capacitance_profile_j_k",
        )
    )


def _heat_capacity_from_mass(cfg: dict[str, Any]) -> float | None:
    mass = cfg.get("solid_mass_kg", cfg.get("matrix_mass_kg", cfg.get("metal_mass_kg")))
    cp = cfg.get("solid_cp_j_kg_k", cfg.get("matrix_cp_j_kg_k", cfg.get("metal_cp_j_kg_k")))
    if mass is None or cp is None:
        return None
    capacitance = max(float(mass), 0.0) * max(float(cp), 0.0)
    return capacitance if capacitance > 0.0 else None


def _distributed_regenerator_outlets_c(hot_in_c: float, cold_in_c: float, solid_profile_c: list[float], effectiveness: float) -> tuple[float, float]:
    eff = float(np.clip(effectiveness, 0.0, 1.0))
    hot_out_c = float(hot_in_c)
    for solid_c in solid_profile_c:
        hot_out_c += eff * (float(solid_c) - hot_out_c)

    cold_out_c = float(cold_in_c)
    for solid_c in reversed(solid_profile_c):
        cold_out_c += eff * (float(solid_c) - cold_out_c)
    return hot_out_c, cold_out_c


def _calibrate_distributed_regenerator_exchange(
    reg_cfg: dict[str, Any],
    air: dict[str, float],
    unknowns: np.ndarray,
    solid_profile_c: list[float],
) -> None:
    if not reg_cfg.get("auto_calibrate_gas_solid_ua_factor", False):
        return

    model = str(reg_cfg.get("model", "")).strip().lower()
    if model in LUMPED_MATRIX_REGENERATOR_MODELS or model in TWO_LUMP_MATRIX_REGENERATOR_MODELS:
        return
    if not solid_profile_c:
        return

    ua_total_w_k = max(float(reg_cfg.get("ua_w_k", 0.0)), 0.0)
    if ua_total_w_k <= 0.0:
        return

    room_c, _, t3_c, t4_target_c, t6_target_c = [float(value) for value in unknowns[:5]]
    span_k = max(abs(t3_c - room_c), 1.0)
    cp_air = max(float(reg_cfg.get("cp_air_j_kg_k", 1005.0)), 1.0e-9)
    capacity_rate_w_k = max(float(air["m_air"]) * cp_air, 1.0e-9)
    eff_min = float(np.clip(float(reg_cfg.get("calibrated_gas_effectiveness_min", 0.0)), 0.0, 0.999999))
    eff_max = float(np.clip(float(reg_cfg.get("calibrated_gas_effectiveness_max", 0.98)), eff_min, 0.999999))

    def objective(effectiveness: float) -> float:
        hot_out_c, cold_out_c = _distributed_regenerator_outlets_c(t3_c, room_c, solid_profile_c, effectiveness)
        hot_error = (hot_out_c - t4_target_c) / span_k
        cold_error = (cold_out_c - t6_target_c) / span_k
        return hot_error * hot_error + cold_error * cold_error

    samples = np.linspace(eff_min, eff_max, 401)
    scores = np.asarray([objective(float(value)) for value in samples], dtype=float)
    best_idx = int(np.argmin(scores))
    lo = float(samples[max(best_idx - 1, 0)])
    hi = float(samples[min(best_idx + 1, samples.size - 1)])
    for _ in range(64):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if objective(left) <= objective(right):
            hi = right
        else:
            lo = left

    effectiveness = float(np.clip(0.5 * (lo + hi), eff_min, eff_max))
    ntu_cell = -float(np.log(max(1.0 - effectiveness, 1.0e-12)))
    factor = ntu_cell * capacity_rate_w_k * max(len(solid_profile_c), 1) / ua_total_w_k
    factor = float(
        np.clip(
            factor,
            max(float(reg_cfg.get("gas_solid_ua_factor_min", 0.0)), 0.0),
            max(float(reg_cfg.get("gas_solid_ua_factor_max", 100.0)), 0.0),
        )
    )
    reg_cfg["gas_solid_ua_factor"] = factor
    reg_cfg["calibrated_gas_effectiveness"] = effectiveness


def _regenerator_initial_profile_c(config: dict[str, Any], unknowns: np.ndarray) -> list[float]:
    reg_cfg = _air_cycle_transient_regenerator_config(config)
    if reg_cfg is None:
        return []
    count = _air_cycle_transient_regenerator_cell_count(config)
    if count <= 0:
        return []

    profile_k = reg_cfg.get("initial_solid_profile_k")
    if isinstance(profile_k, list) and profile_k:
        values = [float(value) - KELVIN_OFFSET for value in profile_k]
    else:
        profile_c = reg_cfg.get("initial_solid_profile_c")
        if isinstance(profile_c, list) and profile_c:
            values = [float(value) for value in profile_c]
        elif "initial_solid_k" in reg_cfg:
            values = [float(reg_cfg["initial_solid_k"]) - KELVIN_OFFSET]
        elif "initial_solid_c" in reg_cfg:
            values = [float(reg_cfg["initial_solid_c"])]
        else:
            room_c, _, t3_c, t4_c, t6_c = [float(value) for value in unknowns[:5]]
            hot_end_c = 0.5 * (t3_c + t6_c)
            cold_end_c = 0.5 * (t4_c + room_c)
            values = np.linspace(hot_end_c, cold_end_c, count).tolist()
    if len(values) >= count:
        return values[:count]
    return values + [values[-1]] * (count - len(values))


def _cascade_dynamic_unknown_index(config: dict[str, Any]) -> int:
    return (
        10
        + (1 if _receiver_enabled(config) else 0)
        + (1 if _lpr_subcooling_control_enabled(config) else 0)
        + (2 if _lpr_inventory_enabled(config) else 0)
    )


def _cascade_exchanger_config(config: dict[str, Any]) -> dict[str, Any] | None:
    vcc_cfg = config.get("vcc_cycle", {})
    cfg = vcc_cfg.get("cascade_exchanger", vcc_cfg.get("cascade_heat_exchanger", {}))
    if not isinstance(cfg, dict) or not cfg.get("enabled", True):
        return None
    model = str(cfg.get("model", "")).strip().lower()
    if model not in LUMPED_CASCADE_EXCHANGER_MODELS:
        return None
    return cfg


def _cascade_exchanger_initial_profile_c(config: dict[str, Any], model: CascadeSystemModel, unknowns: np.ndarray) -> list[float]:
    cfg = _cascade_exchanger_config(config)
    if cfg is None:
        return []
    count = max(1, int(cfg.get("cell_count", cfg.get("cells", 1))))
    profile_k = cfg.get("initial_solid_profile_k", cfg.get("initial_matrix_profile_k"))
    if isinstance(profile_k, list) and profile_k:
        values = [float(value) - KELVIN_OFFSET for value in profile_k]
    else:
        profile_c = cfg.get("initial_solid_profile_c", cfg.get("initial_matrix_profile_c"))
        if isinstance(profile_c, list) and profile_c:
            values = [float(value) for value in profile_c]
        elif "initial_solid_k" in cfg:
            values = [float(cfg["initial_solid_k"]) - KELVIN_OFFSET]
        elif "initial_solid_c" in cfg:
            values = [float(cfg["initial_solid_c"])]
        elif "initial_matrix_k" in cfg:
            values = [float(cfg["initial_matrix_k"]) - KELVIN_OFFSET]
        elif "initial_matrix_c" in cfg:
            values = [float(cfg["initial_matrix_c"])]
        else:
            room_c, _, t3_c, t4_c, t6_c, tevap_c = [float(value) for value in unknowns[:6]]
            air = model._evaluate_air_cycle(room_c + KELVIN_OFFSET, t3_c + KELVIN_OFFSET, t4_c + KELVIN_OFFSET, t6_c + KELVIN_OFFSET)
            ua_total = max(float(cfg.get("ua_w_k", config.get("vcc_cycle", {}).get("cascade_ua_w_k", 0.0))), 0.0)
            ua_air = max(
                float(cfg.get("ua_air_w_k", cfg.get("air_side_ua_w_k", float(cfg.get("air_side_ua_factor", 2.0)) * ua_total))),
                0.0,
            )
            ua_ref = max(
                float(cfg.get("ua_refrigerant_w_k", cfg.get("refrigerant_side_ua_w_k", float(cfg.get("refrigerant_side_ua_factor", 2.0)) * ua_total))),
                1.0e-9,
            )
            cp_air = max(float(cfg.get("cp_air_j_kg_k", 1005.0)), 1.0e-9)
            c_air = max(float(air["m_air"]) * cp_air, 1.0e-9)
            air_effectiveness = 1.0 - float(np.exp(-ua_air / c_air))
            air_conductance = c_air * float(np.clip(air_effectiveness, 0.0, 1.0))
            t2_c = air["t2_k"] - KELVIN_OFFSET
            matrix_c = (air_conductance * t2_c + ua_ref * tevap_c) / max(air_conductance + ua_ref, 1.0e-9)
            low_c = min(tevap_c, t3_c, air["t2_k"] - KELVIN_OFFSET) - 20.0
            high_c = max(tevap_c, t3_c, air["t2_k"] - KELVIN_OFFSET) + 20.0
            values = [float(np.clip(matrix_c, low_c, high_c))]
    if len(values) >= count:
        return values[:count]
    return values + [values[-1]] * (count - len(values))


def _safe_cp_refrigerant_vapor(config: dict[str, Any], tevap_c: float) -> float:
    ref_fluid = config.get("fluids", {}).get("refrigerant", "Ammonia")
    try:
        p_evap = p_sat(tevap_c + KELVIN_OFFSET, ref_fluid)
        return max(float(props_si("C", "P", p_evap, "T", tevap_c + KELVIN_OFFSET + 1.0, ref_fluid)), 1.0)
    except Exception:
        return 2200.0


def _configure_cascade_dynamic_capacitances(config: dict[str, Any], model: CascadeSystemModel, unknowns: np.ndarray) -> None:
    room_c, _, t3_c, t4_c, t6_c, tevap_c = [float(value) for value in unknowns[:6]]
    air = model._evaluate_air_cycle(room_c + KELVIN_OFFSET, t3_c + KELVIN_OFFSET, t4_c + KELVIN_OFFSET, t6_c + KELVIN_OFFSET)
    hx_uas = model._heat_exchanger_uas(air)

    reg_cfg = _air_cycle_transient_regenerator_config(config)
    if reg_cfg is not None:
        if reg_cfg.get("match_nominal_ua", True):
            reg_cfg["ua_w_k"] = float(config.get("air_cycle", {}).get("regenerator_ua_w_k", reg_cfg.get("ua_w_k", 0.0)))
        if not _has_explicit_heat_capacity(reg_cfg) or reg_cfg.get("auto_capacitance", False):
            mass_cap = _heat_capacity_from_mass(reg_cfg)
            if mass_cap is not None:
                reg_cfg["solid_capacitance_j_k"] = mass_cap
            else:
                cp_air = max(float(reg_cfg.get("cp_air_j_kg_k", 1005.0)), 1.0)
                c_air = max(float(air["m_air"]) * cp_air, 1.0e-9)
                tau_s = max(float(reg_cfg.get("auto_capacitance_time_constant_s", 4.5)), 0.0)
                residence_s = max(float(reg_cfg.get("fluid_residence_time_s", 1.0)), 0.0)
                wall_cap = max(float(hx_uas["regenerator"]["ua_w_k"]) * tau_s, 0.0)
                fluid_cap = 2.0 * c_air * residence_s
                reg_cfg["solid_capacitance_j_k"] = max(wall_cap + fluid_cap, 1.0)
        reg_profile_c = _regenerator_initial_profile_c(config, unknowns)
        _calibrate_distributed_regenerator_exchange(reg_cfg, air, unknowns, reg_profile_c)
        reg_cfg["initial_solid_profile_c"] = reg_profile_c

    cascade_cfg = _cascade_exchanger_config(config)
    if cascade_cfg is not None:
        if cascade_cfg.get("match_nominal_ua", True):
            cascade_cfg["ua_w_k"] = float(config.get("vcc_cycle", {}).get("cascade_ua_w_k", cascade_cfg.get("ua_w_k", 0.0)))
        if not _has_explicit_heat_capacity(cascade_cfg) or cascade_cfg.get("auto_capacitance", False):
            mass_cap = _heat_capacity_from_mass(cascade_cfg)
            if mass_cap is not None:
                cascade_cfg["solid_capacitance_j_k"] = mass_cap
            else:
                cp_air = max(float(cascade_cfg.get("cp_air_j_kg_k", 1005.0)), 1.0)
                c_air = max(float(air["m_air"]) * cp_air, 1.0e-9)
                m_ref = max(float(unknowns[7]) + float(unknowns[8]), 1.0e-9)
                if _vcc_evaporators_are_series(config):
                    m_ref = max(_effective_refrigerant_mass_flow(config, float(unknowns[7]), float(unknowns[8])), 1.0e-9)
                c_ref = m_ref * _safe_cp_refrigerant_vapor(config, tevap_c)
                tau_s = max(float(cascade_cfg.get("auto_capacitance_time_constant_s", 5.0)), 0.0)
                air_residence_s = max(float(cascade_cfg.get("air_residence_time_s", 1.0)), 0.0)
                ref_residence_s = max(float(cascade_cfg.get("refrigerant_residence_time_s", 3.0)), 0.0)
                wall_cap = max(float(hx_uas["cascade"]["ua_w_k"]) * tau_s, 0.0)
                fluid_cap = c_air * air_residence_s + c_ref * ref_residence_s
                cascade_cfg["solid_capacitance_j_k"] = max(wall_cap + fluid_cap, 1.0)
        cascade_cfg["initial_solid_profile_c"] = _cascade_exchanger_initial_profile_c(config, model, unknowns)


def _append_cascade_dynamic_states(config: dict[str, Any], model: CascadeSystemModel, unknowns: np.ndarray) -> np.ndarray:
    dynamic_start = _cascade_dynamic_unknown_index(config)
    if int(unknowns.size) > dynamic_start:
        return unknowns

    _configure_cascade_dynamic_capacitances(config, model, unknowns)
    reg_profile = _regenerator_initial_profile_c(config, unknowns)
    cascade_profile = _cascade_exchanger_initial_profile_c(config, model, unknowns)
    extra = reg_profile + cascade_profile
    if not extra:
        return unknowns
    return np.concatenate([np.asarray(unknowns, dtype=float), np.asarray(extra, dtype=float)])


def _expanded_air_cycle_solver_vector(
    config: dict[str, Any],
    configured: list[float] | tuple[float, ...] | None,
    default_base: list[float],
    target_len: int,
    cell_default: float,
    *,
    transient_base_override: list[float] | None = None,
) -> list[float]:
    values = list(configured) if configured is not None else list(default_base)
    if len(values) == target_len:
        return values
    if target_len > 5 and len(values) == 5:
        base = list(transient_base_override) if transient_base_override is not None else values
        return base + [cell_default] * (target_len - 5)
    return values


def _append_lpr_inventory_solver_entries(
    config: dict[str, Any],
    lower: list[float],
    upper: list[float],
    steps: list[float],
    residual_scales: list[float],
    unknowns: np.ndarray,
    *,
    volume_residual_index: int,
) -> None:
    if not _lpr_inventory_enabled(config):
        return

    lpr_cfg = _low_pressure_receiver_config(config)
    residual_scales[volume_residual_index] = float(
        lpr_cfg.get("volume_residual_scale_m3", max(float(lpr_cfg.get("volume_m3", lpr_cfg.get("internal_volume_m3", 0.0))), 1.0e-6))
    )

    idx = len(lower)
    lower.append(float(lpr_cfg.get("liquid_mass_min_kg", 0.0)))
    upper.append(float(lpr_cfg.get("liquid_mass_max_kg", lpr_cfg.get("mass_max_kg", 100.0))))
    steps.append(float(lpr_cfg.get("liquid_mass_step_scale_kg", max(abs(float(unknowns[idx])), 0.01))))
    residual_scales.append(float(lpr_cfg.get("liquid_mass_residual_scale_kg", 0.01)))

    idx = len(lower)
    lower.append(float(lpr_cfg.get("vapor_mass_min_kg", 0.0)))
    upper.append(float(lpr_cfg.get("vapor_mass_max_kg", lpr_cfg.get("mass_max_kg", 100.0))))
    steps.append(float(lpr_cfg.get("vapor_mass_step_scale_kg", max(abs(float(unknowns[idx])), 0.001))))
    residual_scales.append(float(lpr_cfg.get("vapor_mass_residual_scale_kg", 0.001)))


def _append_cascade_dynamic_solver_entries(
    config: dict[str, Any],
    lower: list[float],
    upper: list[float],
    steps: list[float],
    residual_scales: list[float],
    unknowns: np.ndarray,
) -> None:
    reg_cfg = _air_cycle_transient_regenerator_config(config)
    if reg_cfg is not None:
        reg_count = _air_cycle_transient_regenerator_cell_count(config)
        if int(unknowns.size) >= len(lower) + reg_count:
            lower.extend([float(reg_cfg.get("solid_lower_c", -200.0))] * reg_count)
            upper.extend([float(reg_cfg.get("solid_upper_c", 150.0))] * reg_count)
            steps.extend([float(reg_cfg.get("solid_step_scale_k", 10.0))] * reg_count)
            residual_scales.extend([float(reg_cfg.get("solid_residual_scale_k", 1.0))] * reg_count)

    cascade_cfg = _cascade_exchanger_config(config)
    if cascade_cfg is not None:
        cascade_count = max(1, int(cascade_cfg.get("cell_count", cascade_cfg.get("cells", 1))))
        if int(unknowns.size) >= len(lower) + cascade_count:
            lower.extend([float(cascade_cfg.get("solid_lower_c", cascade_cfg.get("matrix_lower_c", -100.0)))] * cascade_count)
            upper.extend([float(cascade_cfg.get("solid_upper_c", cascade_cfg.get("matrix_upper_c", 150.0)))] * cascade_count)
            steps.extend([float(cascade_cfg.get("solid_step_scale_k", cascade_cfg.get("matrix_step_scale_k", 5.0)))] * cascade_count)
            residual_scales.extend([float(cascade_cfg.get("solid_residual_scale_k", cascade_cfg.get("matrix_residual_scale_k", 1.0)))] * cascade_count)


def _configure_cascade_dynamic_residual_scales(config: dict[str, Any], residual_scales: list[float]) -> None:
    reg_cfg = _air_cycle_transient_regenerator_config(config)
    if reg_cfg is not None:
        reg_scale = float(reg_cfg.get("outlet_residual_scale_k", reg_cfg.get("temperature_residual_scale_k", 1.0)))
        residual_scales[3] = max(abs(reg_scale), 1.0e-12)
        residual_scales[4] = max(abs(reg_scale), 1.0e-12)

    cascade_cfg = _cascade_exchanger_config(config)
    if cascade_cfg is not None:
        cascade_scale = float(cascade_cfg.get("outlet_residual_scale_k", cascade_cfg.get("temperature_residual_scale_k", 1.0)))
        residual_scales[5] = max(abs(cascade_scale), 1.0e-12)


def _cascade_step_solver_config(config: dict[str, Any], model: CascadeSystemModel, sim_cfg: dict[str, Any], unknowns: np.ndarray) -> dict[str, Any]:
    lower = [-83.15, -3.15, -100.0, -100.0, -100.0, -73.15, 0.0, 1.0e-6, 1.0e-6, -50.0]
    upper = [46.85, 86.85, 120.0, 120.0, 120.0, 46.85, 90.0, 5.0, 5.0, 46.85]
    steps = [10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 10.0, 0.01, 0.01, 10.0]
    residual_scales = [1.0, 1.0, 1.0, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 0.1, 0.1, 0.1]
    _configure_cascade_dynamic_residual_scales(config, residual_scales)

    if model.high_pressure_receiver_enabled():
        receiver_cfg = _receiver_config(config)
        lower.append(float(receiver_cfg.get("mass_min_kg", sim_cfg.get("receiver_mass_lower_bound_kg", 0.0))))
        upper.append(
            float(receiver_cfg.get("mass_max_kg", receiver_cfg.get("maximum_mass_kg", sim_cfg.get("receiver_mass_upper_bound_kg", 100.0))))
        )
        idx = len(lower) - 1
        steps.append(float(receiver_cfg.get("mass_step_scale_kg", sim_cfg.get("receiver_mass_step_scale_kg", max(abs(float(unknowns[idx])), 0.1)))))
        residual_scales.append(float(receiver_cfg.get("mass_residual_scale_kg", sim_cfg.get("receiver_mass_residual_scale_kg", 0.1))))

    if _lpr_subcooling_control_enabled(config):
        lpr_cfg = _low_pressure_receiver_config(config)
        idx = len(lower)
        lower.append(float(lpr_cfg.get("subcooling_min_k", 0.0)))
        upper.append(float(lpr_cfg.get("subcooling_max_k", 30.0)))
        steps.append(float(lpr_cfg.get("subcooling_step_scale_k", max(abs(float(unknowns[idx])), 1.0))))
        residual_scales.append(float(lpr_cfg.get("subcooling_residual_scale_k", 1.0)))

    _append_lpr_inventory_solver_entries(
        config,
        lower,
        upper,
        steps,
        residual_scales,
        unknowns,
        volume_residual_index=9,
    )
    _append_cascade_dynamic_solver_entries(config, lower, upper, steps, residual_scales, unknowns)

    if len(lower) != int(unknowns.size):
        return sim_cfg

    return {
        **sim_cfg,
        "state_lower_bounds": lower,
        "state_upper_bounds": upper,
        "state_step_scales": steps,
        "residual_scales": residual_scales,
    }


def air_cycle_initial_vector(config: dict[str, Any]) -> np.ndarray:
    guess = config["initial_guess"]
    air_cfg = config.get("air_cycle", {})
    water_loop_cfg = air_cfg.get("water_loop", config.get("water_loop", {}))
    if not isinstance(water_loop_cfg, dict):
        water_loop_cfg = {}
    water_loop_c = float(
        guess.get(
            "water_loop_c",
            water_loop_cfg.get("initial_c", guess.get("sink_c", config.get("boundary_conditions", {}).get("ambient_c", 30.0))),
        )
    )
    return np.array(
        [
            guess["room_c"],
            water_loop_c,
            guess["t3_c"],
            guess["t4_c"],
            guess["t6_c"],
            *_air_cycle_regenerator_initial_cells_c(config, guess),
        ],
        dtype=float,
    )


def vcc_initial_vector(config: dict[str, Any]) -> np.ndarray:
    guess = config["initial_guess"]
    m_ref_total = float(guess["m_ref_kg_s"])
    m_ref_cascade, m_ref_dock = _split_refrigerant_mass_flow(config, m_ref_total)
    values = [
        guess["sink_c"],
        guess["tevap_c"],
        guess["tcond_c"],
        m_ref_cascade,
        m_ref_dock,
    ]
    if _receiver_enabled(config):
        values.append(_receiver_initial_mass_kg(config))
    if _lpr_subcooling_control_enabled(config):
        values.append(_lpr_initial_subcooling_k(config))
    if _lpr_inventory_enabled(config):
        values.extend(_lpr_initial_masses_kg(config))
    return np.array(values, dtype=float)


def _split_refrigerant_mass_flow(config: dict[str, Any], total_m_ref_kg_s: float) -> tuple[float, float]:
    if _vcc_evaporators_are_series(config):
        total_m_ref_kg_s = max(float(total_m_ref_kg_s), 1.0e-6)
        return total_m_ref_kg_s, total_m_ref_kg_s
    valves = config.get("vcc_cycle", {}).get("expansion_valves", {})
    cascade = valves.get("cascade", {})
    dock = valves.get("dock", {})
    cascade_weight = float(cascade.get("opening", 0.75)) * max(expansion_valve_flow_coefficient(cascade), 1.0e-12)
    dock_weight = float(dock.get("opening", 0.25)) * max(expansion_valve_flow_coefficient(dock), 1.0e-12)
    total_weight = cascade_weight + dock_weight
    if total_weight <= 0.0:
        cascade_fraction = 0.75
    else:
        cascade_fraction = cascade_weight / total_weight
    total_m_ref_kg_s = max(float(total_m_ref_kg_s), 2.0e-6)
    m_ref_cascade = max(total_m_ref_kg_s * cascade_fraction, 1.0e-6)
    m_ref_dock = max(total_m_ref_kg_s - m_ref_cascade, 1.0e-6)
    return m_ref_cascade, m_ref_dock


def _state_value(unknowns: np.ndarray, key: str, config: dict[str, Any] | None = None) -> float:
    if key == "m_ref_kg_s":
        if config is not None and _vcc_evaporators_are_series(config):
            return float(
                0.5
                * (
                    unknowns[STATE_INDEX["m_ref_cascade_kg_s"]]
                    + unknowns[STATE_INDEX["m_ref_dock_kg_s"]]
                )
            )
        return float(unknowns[STATE_INDEX["m_ref_cascade_kg_s"]] + unknowns[STATE_INDEX["m_ref_dock_kg_s"]])
    return float(unknowns[STATE_INDEX[key]])


def _set_state_value(config: dict[str, Any], unknowns: np.ndarray, key: str, value: float) -> None:
    if key == "m_ref_kg_s":
        if _vcc_evaporators_are_series(config):
            value = max(float(value), 1.0e-6)
            unknowns[STATE_INDEX["m_ref_cascade_kg_s"]] = value
            unknowns[STATE_INDEX["m_ref_dock_kg_s"]] = value
            return
        current_total = _state_value(unknowns, key, config)
        if current_total > 0.0:
            cascade_fraction = float(unknowns[STATE_INDEX["m_ref_cascade_kg_s"]]) / current_total
        else:
            cascade, dock = _split_refrigerant_mass_flow(config, float(value))
            cascade_fraction = cascade / max(cascade + dock, 1.0e-12)
        value = max(float(value), 2.0e-6)
        unknowns[STATE_INDEX["m_ref_cascade_kg_s"]] = max(value * cascade_fraction, 1.0e-6)
        unknowns[STATE_INDEX["m_ref_dock_kg_s"]] = max(value - unknowns[STATE_INDEX["m_ref_cascade_kg_s"]], 1.0e-6)
        return
    unknowns[STATE_INDEX[key]] = float(value)


def _startup_target_state(config: dict[str, Any], targets: dict[str, float]) -> np.ndarray:
    unknowns = initial_vector(config)
    for key, value in targets.items():
        if key in STATE_INDEX or key == "m_ref_kg_s":
            _set_state_value(config, unknowns, key, float(value))
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
    model = str(infiltration_cfg.get("model", mode)).lower()
    if model in {"tian", "tian_unsteady", "tian_analytical"}:
        return
    if mode == "fixed_w":
        infiltration_cfg["resolved_magnitude_w"] = float(infiltration_cfg.get("magnitude_w", 0.0))
    elif mode == "percent_of_room_load":
        percentage = float(infiltration_cfg.get("load_percentage", 0.10))
        reference_path = infiltration_cfg.get("reference_load_path", "boundary_conditions.load_before_w")
        reference_load_w = float(get_path(config, reference_path))
        infiltration_cfg["resolved_magnitude_w"] = percentage * reference_load_w
    else:
        raise ValueError(f"Unsupported infiltration magnitude_mode: {mode}")


def _free_parameter_bounds(
    startup_cfg: dict[str, Any],
    path: str,
    default: tuple[float, float],
    *,
    clamp_air_performance_speed: bool = False,
) -> tuple[float, float]:
    for item in _startup_free_parameters(startup_cfg):
        if item["path"] == path:
            lower = float(item.get("min", default[0]))
            upper = float(item.get("max", default[1]))
            break
    else:
        lower, upper = default
    if clamp_air_performance_speed and path == "air_cycle.compressor_mass_flow.speed_rpm":
        valid_lower, valid_upper = AIR_PERFORMANCE_MAP_VALIDITY["N_rpm"]
        lower = max(float(lower), float(valid_lower))
        upper = min(float(upper), float(valid_upper))
        if lower > upper:
            lower, upper = float(valid_lower), float(valid_upper)
    return lower, upper


def _air_compressor_speed_controls_mass_flow(config: dict[str, Any]) -> bool:
    model = config["air_cycle"]["compressor_mass_flow"].get("model", "polynomial_volumetric_flow_head")
    return model == "polynomial_volumetric_flow_head_speed" or air_performance_map_model(model)


def _air_compressor_uses_damper_resistance(config: dict[str, Any]) -> bool:
    model = config["air_cycle"]["compressor_mass_flow"].get("model", "polynomial_volumetric_flow_head")
    return model == "polynomial_volumetric_flow_head_speed_damper"


def _air_compressor_uses_constant_mass_flow(config: dict[str, Any]) -> bool:
    model = config["air_cycle"]["compressor_mass_flow"].get("model", "polynomial_volumetric_flow_head")
    if air_performance_map_model(model):
        return False
    return model in {
        "polynomial_volumetric_flow_head_speed_constant_mass_flow",
        "lumped_screw_compressor",
        "lumped_screw_constant_mass_flow",
    }


def _air_pressure_ratio_is_derived(config: dict[str, Any]) -> bool:
    model = config["air_cycle"]["compressor_mass_flow"].get("model", "polynomial_volumetric_flow_head")
    if air_performance_map_model(model):
        return False
    return model in {
        "polynomial_volumetric_flow_head_speed_constant_mass_flow",
        "polynomial_volumetric_flow_head_speed_damper",
    }


def _set_and_mirror_room_load(config: dict[str, Any], value_w: float) -> None:
    mirror_room_load = config["boundary_conditions"].get("load_after_w") == config["boundary_conditions"].get("load_before_w")
    config["boundary_conditions"]["load_before_w"] = float(value_w)
    if mirror_room_load:
        config["boundary_conditions"]["load_after_w"] = float(value_w)


def _backcalculate_dock_evaporator_design(config: dict[str, Any], unknowns: np.ndarray) -> None:
    dock_cfg = config["vcc_cycle"].get("dock_evaporator")
    if not isinstance(dock_cfg, dict):
        return
    model = str(dock_cfg.get("model", "")).lower()
    if model not in {"ua_lmtd_air", "air_lmtd", "lmtd_air"}:
        return
    if not dock_cfg.get("startup_backcalculate_ua_w_k", dock_cfg.get("backcalculate_ua_from_startup", True)):
        return

    _, _, _, _, _, tevap_c, _, _, _, dock_c = unknowns[:10]
    design_air_m_dot = float(dock_cfg.get("design_air_m_dot_kg_s", dock_cfg.get("air_m_dot_kg_s", 3.5)))
    air_m_dot_min = float(dock_cfg.get("air_m_dot_min_kg_s", 0.0))
    air_m_dot_max = float(dock_cfg.get("air_m_dot_max_kg_s", max(design_air_m_dot, air_m_dot_min)))
    air_m_dot_max = max(air_m_dot_max, air_m_dot_min)
    design_air_m_dot = float(np.clip(design_air_m_dot, air_m_dot_min, air_m_dot_max))
    cp_air = float(dock_cfg.get("air_cp_j_kg_k", CP_DOCK_AIR))
    mcp = design_air_m_dot * cp_air
    if mcp <= 0.0:
        raise RuntimeError("Dock evaporator design air mass flow must be positive for startup UA back-calculation.")

    q_design_w = float(config["boundary_conditions"]["dock_load_before_w"])
    air_outlet_c = dock_c - q_design_w / mcp
    if air_outlet_c <= tevap_c:
        raise RuntimeError(
            "Dock evaporator design air flow is too low for the startup dock load: "
            f"air outlet would be {air_outlet_c:.3f} C at refrigerant evaporating temperature {tevap_c:.3f} C. "
            "Increase vcc_cycle.dock_evaporator.design_air_m_dot_kg_s."
        )
    lmtd_k = positive_lmtd(dock_c - tevap_c, air_outlet_c - tevap_c)
    ua_w_k = q_design_w / max(lmtd_k, 1.0e-9)

    dock_cfg["air_m_dot_kg_s"] = design_air_m_dot
    dock_cfg["ua_w_k"] = float(ua_w_k)
    dock_cfg["startup_design_air_outlet_c"] = float(air_outlet_c)
    dock_cfg["startup_design_lmtd_k"] = float(lmtd_k)
    dock_cfg["startup_design_q_dock_w"] = float(q_design_w)


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
    room_c, _, t3_c, t4_c, t6_c, _, _, _, _, _ = unknowns[:10]
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
    room_c, _, t3_c, t4_c, t6_c, _, _, _, _, _ = unknowns[:10]
    speed_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "air_cycle.compressor_mass_flow.speed_rpm",
        (float(speed_cfg.get("speed_rpm", 15000.0)), float(speed_cfg.get("speed_rpm", 15000.0))),
        clamp_air_performance_speed=air_performance_map_model(speed_cfg.get("model", "")),
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


def _solve_air_speed_for_t5(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    t5_target_c: float,
) -> int:
    room_c, _, t3_c, t4_c, t6_c, _, _, _, _, _ = unknowns[:10]
    speed_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "air_cycle.compressor_mass_flow.speed_rpm",
        (float(speed_cfg.get("speed_rpm", 15000.0)), float(speed_cfg.get("speed_rpm", 15000.0))),
        clamp_air_performance_speed=air_performance_map_model(speed_cfg.get("model", "")),
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
        return np.array([(air["t5_k"] - KELVIN_OFFSET - t5_target_c) / 5.0], dtype=float)

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
        raise RuntimeError(f"Could not solve air speed for target t5: {result.message}")
    speed_cfg["speed_rpm"] = float(result.x[0])
    final_error_c = float(residual(result.x)[0] * 5.0)
    if abs(final_error_c) > float(startup_cfg.get("temperature_target_tolerance_c", 1.0e-3)):
        raise RuntimeError(
            f"Could not meet target t5 {t5_target_c:.3f} C; "
            f"best residual is {final_error_c:.3f} C at {float(result.x[0]):.3f} rpm."
        )
    return int(result.nfev)


def _solve_air_speed_for_evaporator_capacity(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    target_capacity_w: float,
) -> int:
    room_c, _, t3_c, t4_c, t6_c, tevap_c, _, _, _, dock_c = unknowns[:10]
    speed_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "air_cycle.compressor_mass_flow.speed_rpm",
        (float(speed_cfg.get("speed_rpm", 15000.0)), float(speed_cfg.get("speed_rpm", 15000.0))),
        clamp_air_performance_speed=air_performance_map_model(speed_cfg.get("model", "")),
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
        q_dock = model._evaluate_dock_evaporator(dock_c, tevap_c)["q_w"]
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
    default_tolerance_w = 1.0
    if air_performance_map_model(speed_cfg.get("model", "")):
        default_tolerance_w = max(default_tolerance_w, 0.02 * abs(float(target_capacity_w)))
    if abs(final_residual_w) > float(startup_cfg.get("capacity_target_tolerance_w", default_tolerance_w)):
        raise RuntimeError(
            f"Could not meet target evaporator capacity {target_capacity_w:.3f} W; "
            f"best residual is {final_residual_w:.3f} W at {float(result.x[0]):.3f} rpm."
        )
    return int(result.nfev)


def _air_metrics_for_head(
    config: dict[str, Any],
    model: CascadeSystemModel,
    room_c: float,
    t3_c: float,
    t4_c: float,
    t6_c: float,
    head_m: float,
) -> dict[str, float]:
    air_cfg = config["air_cycle"]
    p1 = float(air_cfg["p_low_pa"])
    room_k = room_c + KELVIN_OFFSET
    t3_k = t3_c + KELVIN_OFFSET
    t4_k = t4_c + KELVIN_OFFSET
    t6_k = t6_c + KELVIN_OFFSET
    humidity_cfg = air_cfg.get("humid_air", {})
    x_room = humidity_cfg.get("humidity_ratio")
    if x_room is None:
        relative_humidity = float(humidity_cfg.get("room_relative_humidity", 1.0))
        x_room = saturated_room_humidity_ratio(room_k, p1, relative_humidity)
    x_room = float(x_room)

    state1 = humid_air_state(t6_k, p1, x_room)
    compressor_eta_is = float(air_cfg["compressor_eta_is"])
    compressor_head_actual = float(head_m) * 9.80665
    p2 = model._pressure_from_isentropic_head(
        p1,
        x_room,
        state1.entropy_j_kg_da_k,
        state1.enthalpy_j_kg_da,
        compressor_head_actual * compressor_eta_is,
    )
    h2 = state1.enthalpy_j_kg_da + compressor_head_actual
    state2 = state_at_enthalpy(p2, x_room, h2)
    state3 = humid_air_state(t3_k, p2, x_room)
    state4 = humid_air_state(t4_k, p2, x_room)
    state5s = state_at_entropy(p1, x_room, state4.entropy_j_kg_da_k)
    h5 = turbine_actual_enthalpy(state4.enthalpy_j_kg_da, state5s.enthalpy_j_kg_da, air_cfg["turbine_eta_is"])
    state5 = state_at_enthalpy(p1, x_room, h5)
    return {
        "pressure_ratio": p2 / p1,
        "t2_c": state2.temperature_k - KELVIN_OFFSET,
        "t5_c": state5.temperature_k - KELVIN_OFFSET,
        "h2_minus_h3": h2 - state3.enthalpy_j_kg_da,
        "rho1": state1.dry_air_density_kg_m3,
    }


def _solve_air_head_for_t5(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    t5_target_c: float,
) -> tuple[float, dict[str, float], int]:
    room_c, _, t3_c, t4_c, t6_c, _, _, _, _, _ = unknowns[:10]
    flow_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower = float(flow_cfg["head_min_m"])
    upper = float(flow_cfg["head_max_m"])

    def residual_c(head_m: float) -> float:
        metrics = _air_metrics_for_head(config, model, room_c, t3_c, t4_c, t6_c, float(head_m))
        return metrics["t5_c"] - t5_target_c

    sample_heads = np.linspace(lower, upper, int(startup_cfg.get("air_head_scan_points", 400)))
    sample_residuals = np.array([residual_c(float(head)) for head in sample_heads], dtype=float)
    best_idx = int(np.argmin(np.abs(sample_residuals)))
    bracket: tuple[float, float] | None = None
    iterations = len(sample_heads)

    for idx in range(len(sample_heads) - 1):
        f_lo = float(sample_residuals[idx])
        f_hi = float(sample_residuals[idx + 1])
        if abs(f_lo) <= float(startup_cfg.get("temperature_target_tolerance_c", 1.0e-3)):
            head_m = float(sample_heads[idx])
            metrics = _air_metrics_for_head(config, model, room_c, t3_c, t4_c, t6_c, head_m)
            return head_m, metrics, iterations
        if f_lo * f_hi <= 0.0:
            bracket = (float(sample_heads[idx]), float(sample_heads[idx + 1]))
            break

    if bracket is None:
        head_m = float(sample_heads[best_idx])
        metrics = _air_metrics_for_head(config, model, room_c, t3_c, t4_c, t6_c, head_m)
        final_error_c = metrics["t5_c"] - t5_target_c
        raise RuntimeError(
            f"Could not bracket target t5 {t5_target_c:.3f} C over compressor head range "
            f"{lower:.3f} to {upper:.3f} m; best residual is {final_error_c:.3f} C "
            f"at {head_m:.3f} m compressor head."
        )

    lo, hi = bracket
    f_lo = residual_c(lo)
    tolerance_c = float(startup_cfg.get("temperature_target_tolerance_c", 1.0e-3))
    for _ in range(int(startup_cfg.get("air_head_bisection_max_iter", 100))):
        iterations += 1
        mid = 0.5 * (lo + hi)
        f_mid = residual_c(mid)
        if abs(f_mid) <= tolerance_c or abs(hi - lo) <= 1.0e-7:
            head_m = mid
            metrics = _air_metrics_for_head(config, model, room_c, t3_c, t4_c, t6_c, head_m)
            return head_m, metrics, iterations
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    head_m = 0.5 * (lo + hi)
    metrics = _air_metrics_for_head(config, model, room_c, t3_c, t4_c, t6_c, head_m)
    final_error_c = metrics["t5_c"] - t5_target_c
    if abs(final_error_c) > tolerance_c:
        raise RuntimeError(
            f"Could not meet target t5 {t5_target_c:.3f} C after bisection; "
            f"best residual is {final_error_c:.3f} C at {head_m:.3f} m compressor head."
        )
    return head_m, metrics, iterations


def _solve_air_speed_for_head_and_flow(
    config: dict[str, Any],
    startup_cfg: dict[str, Any],
    head_m: float,
    target_volumetric_flow_m3_s: float,
) -> int:
    flow_cfg = config["air_cycle"]["compressor_mass_flow"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "air_cycle.compressor_mass_flow.speed_rpm",
        (float(flow_cfg.get("speed_rpm", 15000.0)), float(flow_cfg.get("speed_rpm", 15000.0))),
        clamp_air_performance_speed=air_performance_map_model(flow_cfg.get("model", "")),
    )

    def residual_q(speed_rpm: float) -> float:
        flow_cfg["speed_rpm"] = float(speed_rpm)
        q_m3_s = volumetric_flow_from_head(flow_cfg, head_m)
        return q_m3_s - target_volumetric_flow_m3_s

    sample_speeds = np.linspace(lower, upper, int(startup_cfg.get("air_speed_scan_points", 400)))
    sample_residuals = np.array([residual_q(float(speed)) for speed in sample_speeds], dtype=float)
    best_idx = int(np.argmin(np.abs(sample_residuals)))
    bracket: tuple[float, float] | None = None
    iterations = len(sample_speeds)
    tolerance_q = float(startup_cfg.get("air_flow_target_tolerance_m3_s", 1.0e-5))

    for idx in range(len(sample_speeds) - 1):
        f_lo = float(sample_residuals[idx])
        f_hi = float(sample_residuals[idx + 1])
        if abs(f_lo) <= tolerance_q:
            flow_cfg["speed_rpm"] = float(sample_speeds[idx])
            return iterations
        if f_lo * f_hi <= 0.0:
            bracket = (float(sample_speeds[idx]), float(sample_speeds[idx + 1]))
            break

    if bracket is None:
        speed_rpm = float(sample_speeds[best_idx])
        flow_cfg["speed_rpm"] = speed_rpm
        final_residual = residual_q(speed_rpm)
        raise RuntimeError(
            f"Could not bracket target air volumetric flow {target_volumetric_flow_m3_s:.6f} m3/s "
            f"over speed range {lower:.3f} to {upper:.3f} rpm; best residual is "
            f"{final_residual:.6f} m3/s at {speed_rpm:.3f} rpm."
        )

    lo, hi = bracket
    f_lo = residual_q(lo)
    for _ in range(int(startup_cfg.get("air_speed_bisection_max_iter", 100))):
        iterations += 1
        mid = 0.5 * (lo + hi)
        f_mid = residual_q(mid)
        if abs(f_mid) <= tolerance_q or abs(hi - lo) <= 1.0e-7:
            flow_cfg["speed_rpm"] = float(mid)
            return iterations
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    speed_rpm = 0.5 * (lo + hi)
    flow_cfg["speed_rpm"] = float(speed_rpm)
    final_residual = residual_q(speed_rpm)
    if abs(final_residual) > tolerance_q:
        raise RuntimeError(
            f"Could not meet target air volumetric flow {target_volumetric_flow_m3_s:.6f} m3/s; "
            f"best residual is {final_residual:.6f} m3/s at {speed_rpm:.3f} rpm."
        )
    return iterations


def _initialize_air_damper_for_evaporator_capacity(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    t5_target_c: float,
    target_capacity_w: float,
) -> int:
    _, _, _, _, _, tevap_c, _, _, _, dock_c = unknowns[:10]
    flow_cfg = config["air_cycle"]["compressor_mass_flow"]
    q_dock = model._evaluate_dock_evaporator(dock_c, tevap_c)["q_w"]
    target_q_cascade_w = float(target_capacity_w) - q_dock
    if target_q_cascade_w <= 0.0:
        raise RuntimeError(
            f"Target evaporator capacity {target_capacity_w:.3f} W is not above dock load {q_dock:.3f} W."
        )

    head_m, metrics, iterations = _solve_air_head_for_t5(config, model, startup_cfg, unknowns, t5_target_c)
    target_m_air_kg_s = target_q_cascade_w / max(metrics["h2_minus_h3"], 1.0e-9)
    target_q_m3_s = target_m_air_kg_s / max(metrics["rho1"], 1.0e-9)
    iterations += _solve_air_speed_for_head_and_flow(config, startup_cfg, head_m, target_q_m3_s)

    opening = float(flow_cfg.get("startup_damper_opening", flow_cfg.get("damper_opening", 0.5)))
    opening_min = float(flow_cfg.get("damper_opening_min", 0.05))
    opening_max = float(flow_cfg.get("damper_opening_max", 1.0))
    opening = min(max(opening, opening_min), opening_max)
    static_pressure_drop_pa = float(
        flow_cfg.get("startup_static_pressure_drop_pa", startup_cfg.get("air_damper_static_pressure_drop_pa", 250.0))
    )
    static_head_m = max(static_pressure_drop_pa, 0.0) / max(metrics["rho1"] * 9.80665, 1.0e-9)
    static_head_m = min(static_head_m, 0.95 * head_m)
    resistance = max(head_m - static_head_m, 1.0e-9) / max((target_q_m3_s / max(opening, 1.0e-9)) ** 2, 1.0e-12)

    flow_cfg["damper_opening"] = opening
    flow_cfg["system_static_head_m"] = float(static_head_m)
    flow_cfg["damper_resistance_head_coefficient"] = float(resistance)
    flow_cfg["operating_head_m"] = float(head_m)
    flow_cfg["startup_target_m_dot_kg_s"] = float(target_m_air_kg_s)
    flow_cfg["startup_target_q_cascade_w"] = float(target_q_cascade_w)
    return iterations


def _initialize_constant_air_flow_for_evaporator_capacity(
    config: dict[str, Any],
    model: CascadeSystemModel,
    startup_cfg: dict[str, Any],
    unknowns: np.ndarray,
    t5_target_c: float,
    target_capacity_w: float,
) -> int:
    _, _, _, _, _, tevap_c, _, _, _, dock_c = unknowns[:10]
    flow_cfg = config["air_cycle"]["compressor_mass_flow"]
    q_dock = model._evaluate_dock_evaporator(dock_c, tevap_c)["q_w"]
    target_q_cascade_w = float(target_capacity_w) - q_dock
    if target_q_cascade_w <= 0.0:
        raise RuntimeError(
            f"Target evaporator capacity {target_capacity_w:.3f} W is not above dock load {q_dock:.3f} W."
        )

    head_m, metrics, iterations = _solve_air_head_for_t5(config, model, startup_cfg, unknowns, t5_target_c)
    target_m_air_kg_s = target_q_cascade_w / max(metrics["h2_minus_h3"], 1.0e-9)
    target_q_m3_s = target_m_air_kg_s / max(metrics["rho1"], 1.0e-9)
    flow_cfg["fixed_m_dot_kg_s"] = float(target_m_air_kg_s)
    flow_cfg["startup_target_q_cascade_w"] = float(target_q_cascade_w)
    iterations += _solve_air_speed_for_head_and_flow(config, startup_cfg, head_m, target_q_m3_s)
    return iterations


def _solve_vcc_speed_for_map_capacity(
    config: dict[str, Any],
    startup_cfg: dict[str, Any],
    tevap_c: float,
    tcond_c: float,
    target_capacity_w: float,
) -> int:
    compressor_cfg = config["vcc_cycle"]["compressor"]
    lower, upper = _free_parameter_bounds(
        startup_cfg,
        "vcc_cycle.compressor.speed_rpm",
        (
            float(compressor_cfg.get("speed_min_rpm", 1226.0)),
            float(compressor_cfg.get("speed_max_rpm", 1610.0)),
        ),
    )

    def residual(speed_rpm: float) -> float:
        compressor_cfg["speed_rpm"] = speed_rpm
        compressor_map = ammonia_compressor_map(
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            speed_rpm,
            check_range=bool(compressor_cfg.get("check_range", False)),
        )
        return compressor_map["Q_W"] - target_capacity_w

    speed_rpm, iterations = _solve_scalar_bisection(residual, lower, upper, tol=1.0e-4)
    compressor_cfg["speed_rpm"] = float(speed_rpm)
    return iterations


def _compressor_displacement_m3_per_rev(compressor_cfg: dict[str, Any]) -> float:
    displacement = float(compressor_cfg.get("displacement_m3_per_rev", 0.0))
    if displacement > 0.0:
        return displacement
    swept_volume_m3_h = compressor_cfg.get("displacement_m3_h", compressor_cfg.get("swept_volume_m3_h"))
    nominal_speed_rpm = compressor_cfg.get("nominal_speed_rpm", compressor_cfg.get("design_speed_rpm"))
    if swept_volume_m3_h is None or nominal_speed_rpm is None:
        return 0.0
    displacement = float(swept_volume_m3_h) / max(float(nominal_speed_rpm) * 60.0, 1.0e-12)
    compressor_cfg["displacement_m3_per_rev"] = float(displacement)
    return displacement


def _initialize_screw_compressor_from_pressure_ratio(
    config: dict[str, Any],
    compressor_cfg: dict[str, Any],
    p_evap: float,
    p_cond: float,
    h7_target: float,
    target_m_ref_kg_s: float,
) -> float:
    pressure_ratio = p_cond / max(p_evap, 1.0e-9)
    screw_map = screw_compressor_pressure_ratio_map(compressor_cfg, pressure_ratio)
    compressor_cfg["eta_is"] = float(screw_map["eta_is"])
    compressor_cfg["eta_v"] = float(screw_map["eta_v"])
    compressor_cfg["q_oil_w"] = float(screw_map["q_oil_w"])

    suction_density = props_si("D", "P", p_evap, "H", h7_target, config["fluids"]["refrigerant"])
    displacement = _compressor_displacement_m3_per_rev(compressor_cfg)
    if displacement <= 0.0 and compressor_cfg.get("startup_backcalculate_displacement", False):
        speed_rps = max(float(compressor_cfg.get("speed_rpm", 0.0)) / 60.0, 1.0e-12)
        displacement = float(target_m_ref_kg_s) / max(suction_density * speed_rps * screw_map["eta_v"], 1.0e-12)
        compressor_cfg["displacement_m3_per_rev"] = displacement
    return compressor_mass_flow_positive_displacement(
        suction_density,
        float(compressor_cfg.get("speed_rpm", 0.0)),
        displacement,
        screw_map["eta_v"],
    )


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
    m_ref_cascade, m_ref_dock = _split_refrigerant_mass_flow(config, m_ref_kg_s)
    unknowns = [room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock, dock_c]
    if _receiver_enabled(config):
        unknowns.append(_receiver_initial_mass_kg(config))
    if _lpr_subcooling_control_enabled(config):
        unknowns.append(_lpr_initial_subcooling_k(config))
    if _lpr_inventory_enabled(config):
        unknowns.extend(_lpr_initial_masses_kg(config))

    return (
        np.array(unknowns, dtype=float),
        t5_target_c,
    )


def _solve_paper_design_initialization(config: dict[str, Any], model: CascadeSystemModel, startup_cfg: dict[str, Any], time_s: float) -> StartupInitializationResult:
    unknowns, t5_target_c = _build_paper_design_unknowns(config, startup_cfg)
    _backcalculate_dock_evaporator_design(config, unknowns)
    target_capacity_w = startup_cfg.get("target_vcc_cooling_capacity_w")
    target_m_air = startup_cfg.get("target_air_mass_flow_kg_s")
    fixed_vcc_speed_rpm = startup_cfg.get("fixed_vcc_compressor_speed_rpm")
    vcc_speed_solved = False
    if _air_compressor_uses_constant_mass_flow(config) and target_capacity_w is not None:
        iteration_count = _initialize_constant_air_flow_for_evaporator_capacity(
            config,
            model,
            startup_cfg,
            unknowns,
            t5_target_c,
            float(target_capacity_w),
        )
        air_speed_solved = True
    elif _air_compressor_uses_damper_resistance(config) and target_capacity_w is not None:
        iteration_count = _initialize_air_damper_for_evaporator_capacity(
            config,
            model,
            startup_cfg,
            unknowns,
            t5_target_c,
            float(target_capacity_w),
        )
        air_speed_solved = True
    elif _air_pressure_ratio_is_derived(config):
        iteration_count = _solve_air_speed_for_t5(config, model, startup_cfg, unknowns, t5_target_c)
        air_speed_solved = True
    else:
        iteration_count = _solve_air_pressure_ratio_for_t5(config, model, startup_cfg, unknowns, t5_target_c)
        air_speed_solved = False

    if target_capacity_w is not None and _air_compressor_speed_controls_mass_flow(config):
        iteration_count += _solve_air_speed_for_evaporator_capacity(config, model, startup_cfg, unknowns, float(target_capacity_w))
        air_speed_solved = True
    elif target_m_air is not None and _air_compressor_speed_controls_mass_flow(config):
        iteration_count += _solve_air_speed_for_mass_flow(config, model, startup_cfg, unknowns, float(target_m_air))
        air_speed_solved = True

    room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, _, _, dock_c = unknowns[:10]
    air = model._evaluate_air_cycle(room_c + KELVIN_OFFSET, t3_c + KELVIN_OFFSET, t4_c + KELVIN_OFFSET, t6_c + KELVIN_OFFSET)
    q_dock = model._evaluate_dock_evaporator(dock_c, tevap_c)["q_w"]
    q_evap_total = air["q_cascade"] + q_dock

    ref_fluid = config["fluids"]["refrigerant"]
    vcc_cfg = config["vcc_cycle"]
    p_evap = p_sat(tevap_c + KELVIN_OFFSET, ref_fluid)
    p_cond_from_tcond = p_sat(tcond_c + KELVIN_OFFSET, ref_fluid)
    p_cond = p_cond_from_tcond
    subcooling_k = _lpr_initial_subcooling_k(config) if _lpr_subcooling_control_enabled(config) else float(vcc_cfg["subcooling_k"])
    lpr_liquid_mass_kg, lpr_vapor_mass_kg = _lpr_initial_masses_kg(config) if _lpr_inventory_enabled(config) else (None, None)
    h9 = h_refrigerant_liquid(tcond_c + KELVIN_OFFSET - subcooling_k, p_cond_from_tcond, ref_fluid, subcooling_k)
    receiver_mass_kg = _receiver_initial_mass_kg(config) if _receiver_enabled(config) else None
    receiver = model._evaluate_high_pressure_receiver(receiver_mass_kg, p_cond, tcond_c + KELVIN_OFFSET, h9)
    h10 = receiver["outlet_enthalpy_j_kg"]
    superheat_target_k = float(startup_cfg.get("superheat_target_k", 1.0))
    if _compressor_uses_saturated_suction(config) or _lpr_inventory_enabled(config):
        h7_target = props_si("H", "P", p_evap, "Q", 1.0, ref_fluid)
        s7_target = props_si("S", "P", p_evap, "Q", 1.0, ref_fluid)
        superheat_target_k = 0.0
    else:
        h7_target = props_si("H", "T", tevap_c + KELVIN_OFFSET + superheat_target_k, "P", p_evap, ref_fluid)
        s7_target = props_si("S", "P", p_evap, "H", h7_target, ref_fluid)
    m_ref = q_evap_total / max(h7_target - h10, 1.0e-9)

    compressor_cfg = vcc_cfg["compressor"]
    compressor_model = compressor_cfg.get("model", "simple_isentropic")
    if compressor_model == "bitzer_variable_speed_map":
        if fixed_vcc_speed_rpm is None:
            iteration_count += _solve_vcc_speed_for_map_capacity(config, startup_cfg, tevap_c, tcond_c, q_evap_total)
            vcc_speed_solved = True
        else:
            compressor_cfg["speed_rpm"] = float(fixed_vcc_speed_rpm)
        compressor_map = ammonia_compressor_map(
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            compressor_cfg["speed_rpm"],
            check_range=bool(compressor_cfg.get("check_range", False)),
        )
        m_ref = compressor_map["mdot_kg_s"]
        compressor_eta_is = ammonia_compressor_eta_is(
            compressor_cfg,
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            compressor_cfg["speed_rpm"],
            ref_fluid,
        )
        if (
            compressor_uses_map_power_pressure_lift(compressor_cfg)
            and compressor_cfg.get("fit_eta_is_from_startup", True)
            and not compressor_uses_ammonia_eta_is_map(compressor_cfg, ref_fluid)
        ):
            eta_fit = compressor_eta_is_from_map_power(
                p_cond_from_tcond,
                h7_target,
                s7_target,
                m_ref,
                compressor_map["P_W"],
                ref_fluid,
            )
            eta_min = float(compressor_cfg.get("eta_is_min", 0.4))
            eta_max = float(compressor_cfg.get("eta_is_max", 0.95))
            compressor_cfg["eta_is"] = float(np.clip(eta_fit, eta_min, eta_max))
            compressor_eta_is = compressor_cfg["eta_is"]
        if compressor_uses_map_power_pressure_lift(compressor_cfg):
            p_cond = compressor_discharge_pressure_from_power(
                p_evap,
                h7_target,
                s7_target,
                m_ref,
                compressor_map["P_W"],
                compressor_eta_is,
                ref_fluid,
                pressure_ratio_min=float(compressor_cfg.get("pressure_ratio_min", 1.000001)),
                pressure_ratio_max=compressor_cfg.get("pressure_ratio_max"),
            )
    elif compressor_model in SCREW_PRESSURE_RATIO_COMPRESSOR_MODELS:
        if fixed_vcc_speed_rpm is not None:
            compressor_cfg["speed_rpm"] = float(fixed_vcc_speed_rpm)
        m_ref = _initialize_screw_compressor_from_pressure_ratio(
            config,
            compressor_cfg,
            p_evap,
            p_cond,
            h7_target,
            m_ref,
        )
    elif compressor_model in {"positive_displacement_clearance", "positive_displacement"}:
        if fixed_vcc_speed_rpm is not None:
            compressor_cfg["speed_rpm"] = float(fixed_vcc_speed_rpm)
        pressure_ratio = p_cond / max(p_evap, 1.0e-9)
        suction_density = props_si("D", "P", p_evap, "H", h7_target, ref_fluid)
        eta_v = compressor_volumetric_efficiency_clearance(
            pressure_ratio,
            float(compressor_cfg.get("clearance_factor", 0.05)),
            float(compressor_cfg.get("polytropic_exponent", 1.25)),
            float(compressor_cfg.get("eta_v_min", 0.3)),
            float(compressor_cfg.get("eta_v_max", 1.0)),
        )
        if compressor_cfg.get("startup_backcalculate_displacement", compressor_cfg.get("backcalculate_displacement_from_startup", True)):
            speed_rps = max(float(compressor_cfg.get("speed_rpm", 0.0)) / 60.0, 1.0e-12)
            compressor_cfg["displacement_m3_per_rev"] = m_ref / max(suction_density * speed_rps * eta_v, 1.0e-12)
        m_ref = compressor_mass_flow_positive_displacement(
            suction_density,
            float(compressor_cfg.get("speed_rpm", 0.0)),
            float(compressor_cfg["displacement_m3_per_rev"]),
            eta_v,
        )
    else:
        eta_is = ammonia_compressor_eta_is(
            compressor_cfg,
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            float(compressor_cfg.get("speed_rpm", 0.0)),
            ref_fluid,
        )
        h8s = props_si("H", "P", p_cond, "S", s7_target, ref_fluid)
        compressor_cfg["work_w"] = float(m_ref * (h8s - h7_target) / max(eta_is, 1.0e-6))
    if _vcc_evaporators_are_series(config):
        m_ref_cascade = m_ref
        m_ref_dock = m_ref
    else:
        q_total_for_split = max(q_evap_total, 1.0e-9)
        m_ref_cascade = m_ref * air["q_cascade"] / q_total_for_split
        m_ref_dock = m_ref * q_dock / q_total_for_split
    unknowns[STATE_INDEX["m_ref_cascade_kg_s"]] = max(m_ref_cascade, 1.0e-6)
    unknowns[STATE_INDEX["m_ref_dock_kg_s"]] = max(m_ref_dock, 1.0e-6)

    legacy_valve_cfg = dict(vcc_cfg.get("expansion_valve", {}))
    branch_valves = vcc_cfg.setdefault("expansion_valves", {})
    if _vcc_evaporators_are_series(config):
        branch_targets = {
            "cascade": (air["q_cascade"] + q_dock, _effective_refrigerant_mass_flow(config, m_ref_cascade, m_ref_dock)),
        }
    else:
        branch_targets = {
            "cascade": (air["q_cascade"], unknowns[STATE_INDEX["m_ref_cascade_kg_s"]]),
            "dock": (q_dock, unknowns[STATE_INDEX["m_ref_dock_kg_s"]]),
        }
    receiver = model._evaluate_high_pressure_receiver(receiver_mass_kg, p_cond, tcond_c + KELVIN_OFFSET, h9)
    valve_inlet_density = receiver["outlet_density_kg_m3"]
    valve_flow_factor = expansion_valve_flow_factor(p_cond, p_evap, valve_inlet_density)
    for branch, (_, branch_m_ref) in branch_targets.items():
        valve_cfg = branch_valves.setdefault(branch, dict(legacy_valve_cfg))
        had_orifice_coefficient = "flow_coefficient_m2" in valve_cfg or "flow_coefficient_kg_s_sqrt_pa_density" in valve_cfg
        if "flow_coefficient_m2" not in valve_cfg:
            legacy_coefficient = legacy_valve_cfg.get(
                "flow_coefficient_m2",
                legacy_valve_cfg.get("flow_coefficient_kg_s_sqrt_pa_density", legacy_valve_cfg.get("flow_coefficient_kg_s_pa", 0.0)),
            )
            valve_cfg.setdefault("flow_coefficient_m2", legacy_coefficient)
        opening_min = float(valve_cfg.get("opening_min", 0.05))
        opening_max = float(valve_cfg.get("opening_max", 1.0))
        opening_target = float(np.clip(float(valve_cfg.get("opening", 0.5)), opening_min, opening_max))
        backcalculate_coefficient = bool(
            valve_cfg.get(
                "startup_backcalculate_flow_coefficient_m2",
                not had_orifice_coefficient,
            )
        )
        if backcalculate_coefficient:
            valve_cfg["flow_coefficient_m2"] = float(branch_m_ref / max(opening_target * valve_flow_factor, 1.0e-12))
            valve_cfg["opening"] = opening_target
        else:
            valve_opening = branch_m_ref / max(expansion_valve_flow_coefficient(valve_cfg) * valve_flow_factor, 1.0e-12)
            valve_cfg["opening"] = float(np.clip(valve_opening, opening_min, opening_max))

    ref = model._evaluate_refrigerant_cycle(
        tevap_c + KELVIN_OFFSET,
        tcond_c + KELVIN_OFFSET,
        unknowns[STATE_INDEX["m_ref_cascade_kg_s"]],
        unknowns[STATE_INDEX["m_ref_dock_kg_s"]],
        air["q_cascade"],
        q_dock,
        receiver_mass_kg,
        subcooling_k,
        lpr_liquid_mass_kg,
        lpr_vapor_mass_kg,
    )
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
        if _path_exists(config, path)
        if path != "air_cycle.compressor_mass_flow.speed_rpm" or air_speed_solved
        if path != "air_cycle.pressure_ratio" or not _air_pressure_ratio_is_derived(config)
        if path != "vcc_cycle.compressor.speed_rpm" or vcc_speed_solved
    }
    for path, value in free_parameter_values.items():
        snapshot[f"startup_solved_{path.replace('.', '_')}"] = value
    snapshot["startup_paper_t5_target_c"] = t5_target_c
    snapshot["startup_paper_regenerator_effectiveness"] = float(startup_cfg.get("regenerator_effectiveness", 0.9))
    if target_capacity_w is not None:
        snapshot["startup_paper_target_vcc_cooling_capacity_w"] = float(target_capacity_w)
    if fixed_vcc_speed_rpm is not None:
        snapshot["startup_fixed_vcc_compressor_speed_rpm"] = float(fixed_vcc_speed_rpm)

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
            jacobian_scheme=startup_cfg.get("jacobian_scheme", sim_cfg.get("jacobian_scheme", "central")),
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
                if _path_exists(config, path)
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
        x0.append(_state_value(fixed_unknowns, key, config))
        if key == "receiver_mass_kg":
            receiver_cfg = _receiver_config(config)
            lower.append(float(receiver_cfg.get("mass_min_kg", startup_cfg.get("receiver_mass_min_kg", 0.0))))
            upper.append(float(receiver_cfg.get("mass_max_kg", startup_cfg.get("receiver_mass_max_kg", 100.0))))
            x_scale.append(max(abs(x0[-1]), 0.1))
        elif key.startswith("m_ref"):
            lower.append(float(startup_cfg.get("mass_flow_min_kg_s", 1.0e-6)))
            upper.append(float(startup_cfg.get("mass_flow_max_kg_s", 5.0)))
            x_scale.append(max(abs(x0[-1]), 1.0e-3))
        else:
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
            _set_state_value(config, unknowns, key, x[offset])
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
        lpr_cfg = _low_pressure_receiver_config(config)
        hpr_extra = 1 if model.high_pressure_receiver_enabled() else 0
        subcooling_idx = 10 + hpr_extra if model.lpr_subcooling_control_enabled() else None
        lpr_idx = 10 + hpr_extra + (1 if model.lpr_subcooling_control_enabled() else 0)
        for idx, value in enumerate(balance):
            if model.lpr_inventory_enabled() and idx == 9:
                scale = float(lpr_cfg.get("volume_residual_scale_m3", max(float(lpr_cfg.get("volume_m3", lpr_cfg.get("internal_volume_m3", 0.0))), 1.0e-6)))
            elif subcooling_idx is not None and idx == subcooling_idx:
                scale = scales["delta_t_c"]
            elif model.lpr_inventory_enabled() and idx == lpr_idx:
                scale = float(lpr_cfg.get("liquid_mass_residual_scale_kg", 0.01))
            elif model.lpr_inventory_enabled() and idx == lpr_idx + 1:
                scale = float(lpr_cfg.get("vapor_mass_residual_scale_kg", 0.001))
            else:
                scale = scales["kg_s"] if idx >= 7 else scales["w"]
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
        startup_residual_size += 2

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
            extra_residuals.append((metrics["refrigerant_cascade_superheat_k"] - float(superheat_target_k)) / scales["delta_t_c"])
            extra_residuals.append((metrics["refrigerant_dock_superheat_k"] - float(superheat_target_k)) / scales["delta_t_c"])
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


def initialize_vcc_standalone_design(
    config: dict[str, Any],
    model: CascadeSystemModel,
    unknowns: np.ndarray,
) -> StartupInitializationResult:
    sim_cfg = config["simulation"]
    startup_cfg = sim_cfg.get("startup_initialization", {})
    time_s = float(startup_cfg.get("time_s", sim_cfg["t_start_s"]))
    if not startup_cfg.get("enabled", False):
        snapshot = model.vcc_post_process(unknowns, time_s).values
        return StartupInitializationResult(unknowns, 0, 0.0, snapshot, cache_hit=False, cache_path=None)

    targets = startup_cfg.get("targets", {})
    sink_c = float(targets.get("sink_c", unknowns[0]))
    tevap_c = float(targets.get("tevap_c", unknowns[1]))
    tcond_c = float(targets.get("tcond_c", unknowns[2]))
    unknowns[0] = sink_c
    unknowns[1] = tevap_c
    unknowns[2] = tcond_c

    loads = model._vcc_standalone_loads_w(time_s)
    q_evap_total = max(float(loads["total_w"]), 0.0)
    if q_evap_total <= 0.0:
        snapshot = model.vcc_post_process(unknowns, time_s).values
        return StartupInitializationResult(unknowns, 0, 0.0, snapshot, cache_hit=False, cache_path=None)

    ref_fluid = config["fluids"]["refrigerant"]
    vcc_cfg = config["vcc_cycle"]
    p_evap = p_sat(tevap_c + KELVIN_OFFSET, ref_fluid)
    p_cond = p_sat(tcond_c + KELVIN_OFFSET, ref_fluid)
    subcooling_k = _lpr_initial_subcooling_k(config) if _lpr_subcooling_control_enabled(config) else float(vcc_cfg["subcooling_k"])
    h9 = h_refrigerant_liquid(tcond_c + KELVIN_OFFSET - subcooling_k, p_cond, ref_fluid, subcooling_k)
    receiver_mass_kg = float(unknowns[5]) if _receiver_enabled(config) and unknowns.size > 5 else None
    receiver = model._evaluate_high_pressure_receiver(receiver_mass_kg, p_cond, tcond_c + KELVIN_OFFSET, h9)
    h10 = receiver["outlet_enthalpy_j_kg"]
    superheat_target_k = float(startup_cfg.get("superheat_target_k", vcc_cfg.get("standalone_superheat_target_k", 1.0)))
    if _compressor_uses_saturated_suction(config) or _lpr_inventory_enabled(config):
        h7_target = props_si("H", "P", p_evap, "Q", 1.0, ref_fluid)
        s7_target = props_si("S", "P", p_evap, "Q", 1.0, ref_fluid)
        superheat_target_k = 0.0
    else:
        h7_target = props_si("H", "T", tevap_c + KELVIN_OFFSET + superheat_target_k, "P", p_evap, ref_fluid)
        s7_target = props_si("S", "P", p_evap, "H", h7_target, ref_fluid)
    m_ref = q_evap_total / max(h7_target - h10, 1.0e-9)

    compressor_cfg = vcc_cfg["compressor"]
    compressor_model = compressor_cfg.get("model", "simple_isentropic")
    iteration_count = 0
    fixed_vcc_speed_rpm = startup_cfg.get("fixed_vcc_compressor_speed_rpm")
    if compressor_model == "bitzer_variable_speed_map":
        if fixed_vcc_speed_rpm is None:
            iteration_count += _solve_vcc_speed_for_map_capacity(config, startup_cfg, tevap_c, tcond_c, q_evap_total)
        else:
            compressor_cfg["speed_rpm"] = float(fixed_vcc_speed_rpm)
        compressor_map = ammonia_compressor_map(
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            compressor_cfg["speed_rpm"],
            check_range=bool(compressor_cfg.get("check_range", False)),
        )
        m_ref = compressor_map["mdot_kg_s"]
        compressor_eta_is = ammonia_compressor_eta_is(
            compressor_cfg,
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            compressor_cfg["speed_rpm"],
            ref_fluid,
        )
        if (
            compressor_uses_map_power_pressure_lift(compressor_cfg)
            and compressor_cfg.get("fit_eta_is_from_startup", True)
            and not compressor_uses_ammonia_eta_is_map(compressor_cfg, ref_fluid)
        ):
            eta_fit = compressor_eta_is_from_map_power(p_cond, h7_target, s7_target, m_ref, compressor_map["P_W"], ref_fluid)
            compressor_cfg["eta_is"] = float(
                np.clip(
                    eta_fit,
                    float(compressor_cfg.get("eta_is_min", 0.4)),
                    float(compressor_cfg.get("eta_is_max", 0.95)),
                )
            )
            compressor_eta_is = compressor_cfg["eta_is"]
    elif compressor_model in SCREW_PRESSURE_RATIO_COMPRESSOR_MODELS:
        if fixed_vcc_speed_rpm is not None:
            compressor_cfg["speed_rpm"] = float(fixed_vcc_speed_rpm)
        m_ref = _initialize_screw_compressor_from_pressure_ratio(
            config,
            compressor_cfg,
            p_evap,
            p_cond,
            h7_target,
            m_ref,
        )
    elif compressor_model in {"positive_displacement_clearance", "positive_displacement"}:
        if fixed_vcc_speed_rpm is not None:
            compressor_cfg["speed_rpm"] = float(fixed_vcc_speed_rpm)
        pressure_ratio = p_cond / max(p_evap, 1.0e-9)
        suction_density = props_si("D", "P", p_evap, "H", h7_target, ref_fluid)
        eta_v = compressor_volumetric_efficiency_clearance(
            pressure_ratio,
            float(compressor_cfg.get("clearance_factor", 0.05)),
            float(compressor_cfg.get("polytropic_exponent", 1.25)),
            float(compressor_cfg.get("eta_v_min", 0.3)),
            float(compressor_cfg.get("eta_v_max", 1.0)),
        )
        if compressor_cfg.get("startup_backcalculate_displacement", compressor_cfg.get("backcalculate_displacement_from_startup", True)):
            speed_rps = max(float(compressor_cfg.get("speed_rpm", 0.0)) / 60.0, 1.0e-12)
            compressor_cfg["displacement_m3_per_rev"] = m_ref / max(suction_density * speed_rps * eta_v, 1.0e-12)
        m_ref = compressor_mass_flow_positive_displacement(
            suction_density,
            float(compressor_cfg.get("speed_rpm", 0.0)),
            float(compressor_cfg["displacement_m3_per_rev"]),
            eta_v,
        )
    else:
        eta_is = ammonia_compressor_eta_is(
            compressor_cfg,
            tcond_c + KELVIN_OFFSET,
            tevap_c + KELVIN_OFFSET,
            float(compressor_cfg.get("speed_rpm", 0.0)),
            ref_fluid,
        )
        h8s = props_si("H", "P", p_cond, "S", s7_target, ref_fluid)
        compressor_cfg["work_w"] = float(m_ref * (h8s - h7_target) / max(eta_is, 1.0e-6))

    q_cascade = max(float(loads["cascade_w"]), 0.0)
    q_dock = max(float(loads["dock_w"]), 0.0)
    if _vcc_evaporators_are_series(config):
        m_ref_cascade = m_ref
        m_ref_dock = m_ref
    elif q_cascade + q_dock > 0.0:
        m_ref_cascade = m_ref * q_cascade / max(q_cascade + q_dock, 1.0e-9)
        m_ref_dock = m_ref * q_dock / max(q_cascade + q_dock, 1.0e-9)
    else:
        m_ref_cascade, m_ref_dock = _split_refrigerant_mass_flow(config, m_ref)
    unknowns[3] = max(m_ref_cascade, 1.0e-6)
    unknowns[4] = max(m_ref_dock, 1.0e-6)

    legacy_valve_cfg = dict(vcc_cfg.get("expansion_valve", {}))
    branch_valves = vcc_cfg.setdefault("expansion_valves", {})
    valve_inlet_density = receiver["outlet_density_kg_m3"]
    valve_flow_factor = expansion_valve_flow_factor(p_cond, p_evap, valve_inlet_density)
    if _vcc_evaporators_are_series(config):
        branch_targets = {"cascade": _effective_refrigerant_mass_flow(config, unknowns[3], unknowns[4])}
    else:
        branch_targets = {
            "cascade": unknowns[3],
            "dock": unknowns[4],
        }
    for branch, branch_m_ref in branch_targets.items():
        valve_cfg = branch_valves.setdefault(branch, dict(legacy_valve_cfg))
        had_orifice_coefficient = "flow_coefficient_m2" in valve_cfg or "flow_coefficient_kg_s_sqrt_pa_density" in valve_cfg
        if "flow_coefficient_m2" not in valve_cfg:
            legacy_coefficient = legacy_valve_cfg.get(
                "flow_coefficient_m2",
                legacy_valve_cfg.get("flow_coefficient_kg_s_sqrt_pa_density", legacy_valve_cfg.get("flow_coefficient_kg_s_pa", 0.0)),
            )
            valve_cfg.setdefault("flow_coefficient_m2", legacy_coefficient)
        opening_min = float(valve_cfg.get("opening_min", 0.05))
        opening_max = float(valve_cfg.get("opening_max", 1.0))
        opening_target = float(np.clip(float(valve_cfg.get("opening", 0.5)), opening_min, opening_max))
        backcalculate_coefficient = bool(
            valve_cfg.get(
                "startup_backcalculate_flow_coefficient_m2",
                not had_orifice_coefficient,
            )
        )
        if backcalculate_coefficient:
            valve_cfg["flow_coefficient_m2"] = float(branch_m_ref / max(opening_target * valve_flow_factor, 1.0e-12))
            valve_cfg["opening"] = opening_target
        else:
            valve_opening = branch_m_ref / max(expansion_valve_flow_coefficient(valve_cfg) * valve_flow_factor, 1.0e-12)
            valve_cfg["opening"] = float(np.clip(valve_opening, opening_min, opening_max))

    snapshot = model.vcc_post_process(unknowns, time_s).values
    snapshot["startup_vcc_standalone_superheat_target_k"] = superheat_target_k
    snapshot["startup_solved_vcc_cycle_compressor_displacement_m3_per_rev"] = float(
        compressor_cfg.get("displacement_m3_per_rev", 0.0)
    )
    snapshot["startup_solved_vcc_cycle_expansion_valves_cascade_flow_coefficient_m2"] = float(
        branch_valves.get("cascade", {}).get("flow_coefficient_m2", 0.0)
    )
    snapshot["startup_solved_vcc_cycle_expansion_valves_dock_flow_coefficient_m2"] = float(
        branch_valves.get("dock", {}).get("flow_coefficient_m2", 0.0)
    )
    return StartupInitializationResult(unknowns, iteration_count, 0.0, snapshot, cache_hit=False, cache_path=None)


def solve_dynamic_step(
    residual_fn,
    x0: np.ndarray,
    sim_cfg: dict[str, Any],
    jacobian_executor: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, int]:
    residual0 = np.asarray(residual_fn(np.asarray(x0, dtype=float)), dtype=float)
    configured_lower = sim_cfg.get("state_lower_bounds")
    configured_upper = sim_cfg.get("state_upper_bounds")
    if configured_lower is not None and configured_upper is not None:
        lower = np.asarray(configured_lower, dtype=float)
        upper = np.asarray(configured_upper, dtype=float)
        x0_array = np.asarray(x0, dtype=float)
        if lower.size != x0_array.size or upper.size != x0_array.size:
            raise ValueError("Configured state bounds must match the dynamic state vector length.")
        residual_scales = np.asarray(sim_cfg.get("residual_scales", np.ones_like(residual0)), dtype=float)
        if residual_scales.size != residual0.size:
            residual_scales = np.ones_like(residual0)
        residual_scales = np.maximum(np.abs(residual_scales), 1.0e-12)

        def scaled_residual_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray(residual_fn(x), dtype=float) / residual_scales

        x0_bounded = np.clip(x0_array, lower, upper)
        step_scales = np.asarray(sim_cfg.get("state_step_scales", np.maximum(np.abs(x0_bounded), 1.0)), dtype=float)
        if step_scales.size != x0_bounded.size:
            step_scales = np.maximum(np.abs(x0_bounded), 1.0)
        result = least_squares(
            scaled_residual_fn,
            x0_bounded,
            bounds=(lower, upper),
            x_scale=np.maximum(np.abs(step_scales), 1.0e-12),
            ftol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            xtol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            gtol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            max_nfev=sim_cfg.get("fallback_max_function_evals", 500),
        )
        if result.success:
            return np.asarray(result.x, dtype=float), int(result.nfev)

    if residual0.size in {10, 11} and np.asarray(x0, dtype=float).size in {10, 11}:
        residual_scales = np.ones_like(residual0)
        residual_scales[3:7] = float(sim_cfg.get("fallback_power_residual_scale_w", 1.0e5))
        residual_scales[7:10] = float(sim_cfg.get("fallback_mass_flow_residual_scale_kg_s", 0.1))
        if residual0.size > 10:
            residual_scales[10] = float(sim_cfg.get("receiver_mass_residual_scale_kg", 0.1))

        def scaled_residual_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray(residual_fn(x), dtype=float) / residual_scales

        x0_array = np.asarray(x0, dtype=float)
        lower_values = [-83.15, -3.15, -100.0, -100.0, -100.0, -73.15, 0.0, 1.0e-6, 1.0e-6, -50.0]
        upper_values = [46.85, 86.85, 120.0, 120.0, 120.0, 46.85, 90.0, 5.0, 5.0, 46.85]
        step_values = [10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 10.0, 0.01, 0.01, 10.0]
        if x0_array.size > 10:
            lower_values.append(float(sim_cfg.get("receiver_mass_lower_bound_kg", 0.0)))
            upper_values.append(float(sim_cfg.get("receiver_mass_upper_bound_kg", 100.0)))
            step_values.append(float(sim_cfg.get("receiver_mass_step_scale_kg", max(abs(x0_array[10]), 0.1))))
        lower = np.array(lower_values, dtype=float)
        upper = np.array(upper_values, dtype=float)
        x0_bounded = np.clip(x0_array, lower, upper)
        step_scales = np.array(step_values, dtype=float)
        x = x0_bounded.copy()
        for iteration in range(1, int(sim_cfg.get("newton_max_iter", 100)) + 1):
            residual = scaled_residual_fn(x)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm < float(sim_cfg["newton_tol"]):
                return x, iteration
            jac = np.zeros((residual.size, x.size), dtype=float)
            fd_step = float(sim_cfg["fd_step"])
            def evaluate_jacobian_column(col: int) -> np.ndarray:
                dx = np.zeros_like(x)
                dx[col] = fd_step * max(abs(x[col]), step_scales[col])
                x_plus = np.clip(x + dx, lower, upper)
                actual_dx = x_plus[col] - x[col]
                if abs(actual_dx) <= 1.0e-14:
                    x_minus = np.clip(x - dx, lower, upper)
                    actual_dx = x[col] - x_minus[col]
                    return (residual - scaled_residual_fn(x_minus)) / max(actual_dx, 1.0e-14)
                return (scaled_residual_fn(x_plus) - residual) / actual_dx

            if jacobian_executor is None:
                for col in range(x.size):
                    jac[:, col] = evaluate_jacobian_column(col)
            else:
                for col, column in enumerate(jacobian_executor.map(evaluate_jacobian_column, range(x.size))):
                    jac[:, col] = column
            try:
                delta = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
            damping = 1.0
            while damping > 1.0e-4:
                trial = np.clip(x + damping * delta, lower, upper)
                trial_residual = scaled_residual_fn(trial)
                if float(np.linalg.norm(trial_residual, ord=np.inf)) < residual_norm:
                    x = trial
                    break
                damping *= 0.5
            else:
                break

        result = least_squares(
            scaled_residual_fn,
            x0_bounded,
            bounds=(lower, upper),
            x_scale=np.maximum(np.abs(x0_bounded), step_scales),
            ftol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            xtol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            gtol=sim_cfg.get("fallback_least_squares_tol", sim_cfg["newton_tol"]),
            max_nfev=sim_cfg.get("fallback_max_function_evals", 500),
        )
        if result.success:
            return np.asarray(result.x, dtype=float), int(result.nfev)

    try:
        return newton_raphson_fd(
            residual_fn,
            x0,
            tol=sim_cfg["newton_tol"],
            max_iter=sim_cfg["newton_max_iter"],
            step=sim_cfg["fd_step"],
            jacobian_scheme=sim_cfg.get("jacobian_scheme", "central"),
        )
    except NewtonSolveError:
        residual_scales = np.ones_like(residual0)
        if residual0.size in {10, 11}:
            residual_scales[3:7] = float(sim_cfg.get("fallback_power_residual_scale_w", 1.0e5))
            residual_scales[7:10] = float(sim_cfg.get("fallback_mass_flow_residual_scale_kg_s", 0.1))
            if residual0.size > 10:
                residual_scales[10] = float(sim_cfg.get("receiver_mass_residual_scale_kg", 0.1))

        def scaled_residual_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray(residual_fn(x), dtype=float) / residual_scales

        result = least_squares(
            scaled_residual_fn,
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


def _configure_air_performance_map_controls(config: dict[str, Any]) -> None:
    air_cfg = config.get("air_cycle", {})
    flow_cfg = air_cfg.get("compressor_mass_flow", {})
    if not isinstance(flow_cfg, dict) or not air_performance_map_model(flow_cfg.get("model", "")):
        return

    speed_min, speed_max = AIR_PERFORMANCE_MAP_VALIDITY["N_rpm"]
    flow_cfg.setdefault("speed_min_rpm", float(speed_min))
    flow_cfg.setdefault("speed_max_rpm", float(speed_max))
    flow_cfg["speed_rpm"] = float(np.clip(float(flow_cfg.get("speed_rpm", 15000.0)), speed_min, speed_max))

    control_cfg = config.get("control", {})
    if not isinstance(control_cfg, dict):
        return
    for controller in control_cfg.get("controllers", []):
        if not isinstance(controller, dict):
            continue
        if controller.get("actuator_path") != "air_cycle.compressor_mass_flow.speed_rpm":
            continue
        controller["u_min"] = max(float(controller.get("u_min", speed_min)), speed_min)
        controller["u_max"] = min(float(controller.get("u_max", speed_max)), speed_max)
        if controller["u_min"] > controller["u_max"]:
            controller["u_min"] = float(speed_min)
            controller["u_max"] = float(speed_max)
        if "bias" in controller:
            controller["bias"] = float(np.clip(float(controller["bias"]), controller["u_min"], controller["u_max"]))


def _filtered_control_system(config: dict[str, Any], measurements: dict[str, Any], frozen_actuator_paths: set[str] | None = None) -> ControlSystem:
    _configure_air_performance_map_controls(config)
    control_cfg = config.get("control", {})
    if not control_cfg.get("enabled", False):
        return ControlSystem(config, frozen_actuator_paths=frozen_actuator_paths)

    compatible_controllers = []
    for controller in control_cfg.get("controllers", []):
        measurement = controller.get("measurement")
        if measurement not in measurements:
            continue
        try:
            if not np.isfinite(float(measurements[measurement])):
                continue
        except (TypeError, ValueError):
            continue
        actuator_path = controller.get("actuator_path")
        if not isinstance(actuator_path, str) or not _path_exists(config, actuator_path):
            continue
        compatible_controllers.append(controller)

    if len(compatible_controllers) == len(control_cfg.get("controllers", [])):
        return ControlSystem(config, frozen_actuator_paths=frozen_actuator_paths)

    filtered_config = deepcopy(config)
    filtered_config["control"] = {**control_cfg, "controllers": compatible_controllers, "enabled": bool(compatible_controllers)}
    return ControlSystem(filtered_config, frozen_actuator_paths=frozen_actuator_paths)


def run_simulation(config: dict) -> list[dict[str, float]]:
    plant_config = deepcopy(config)
    configure_property_backend_from_config(plant_config)
    _configure_air_performance_map_controls(plant_config)
    _configure_cascade_lumped_heat_exchangers(plant_config)
    _configure_receiver_defaults(plant_config)
    _configure_lpr_subcooling_defaults(plant_config)
    _configure_lpr_inventory_defaults(plant_config)
    mode = system_mode(plant_config)
    if mode == "air_cycle":
        return _run_air_cycle_simulation(plant_config)
    if mode == "vcc":
        return _run_vcc_simulation(plant_config)
    return _run_cascade_simulation(plant_config)


def _run_cascade_simulation(plant_config: dict[str, Any]) -> list[dict[str, float]]:
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

    unknowns = _append_cascade_dynamic_states(plant_config, model, unknowns)
    model.refresh_dynamic_ua_nominal_flows(unknowns)
    step_sim_cfg = _cascade_step_solver_config(plant_config, model, sim_cfg, unknowns)

    if startup_snapshot:
        for controller in plant_config.get("control", {}).get("controllers", []):
            setpoint_from_startup = controller.get("setpoint_from_startup")
            if setpoint_from_startup and setpoint_from_startup in startup_snapshot:
                controller["setpoint"] = float(startup_snapshot[setpoint_from_startup])
            solved_bias_key = f"startup_solved_{controller['actuator_path'].replace('.', '_')}"
            if controller.get("bias_from_startup", True) and solved_bias_key in startup_snapshot:
                controller["bias"] = float(startup_snapshot[solved_bias_key])

    configure_disturbances(plant_config)
    model.reset_room_moisture(float(unknowns[0]))
    model.reset_infiltration_disturbance(float(unknowns[0]), float(unknowns[9]))
    control = ControlSystem(plant_config, frozen_actuator_paths=frozen_actuator_paths)
    state_values = [unknowns[0], unknowns[1], unknowns[9], model.current_room_humidity_ratio()]
    if model.high_pressure_receiver_enabled():
        state_values.append(float(unknowns[10]))
    if model.lpr_subcooling_control_enabled():
        subcooling_idx = 10 + (1 if model.high_pressure_receiver_enabled() else 0)
        state_values.append(float(unknowns[subcooling_idx]))
    if model.lpr_inventory_enabled():
        lpr_idx = 10 + (1 if model.high_pressure_receiver_enabled() else 0) + (1 if model.lpr_subcooling_control_enabled() else 0)
        state_values.extend([float(unknowns[lpr_idx]), float(unknowns[lpr_idx + 1])])
    dynamic_idx = _cascade_dynamic_unknown_index(plant_config)
    if int(unknowns.size) > dynamic_idx:
        state_values.extend([float(value) for value in unknowns[dynamic_idx:]])
    state = np.array(state_values, dtype=float)
    history: list[dict[str, float]] = []
    jacobian_workers = max(1, int(sim_cfg.get("jacobian_workers", 1)))

    _log(f"[run] steps={len(times) - 1} dt={dt_s:.1f}s interval={progress_interval}")
    jacobian_executor = ThreadPoolExecutor(max_workers=jacobian_workers) if jacobian_workers > 1 else None
    try:
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
                model.reset_receiver_inlet_flow(step.values.get("compressor_map_flow_kg_s", step.values["m_ref_kg_s"]))
                _log(
                    f"[startup] t={time_s / 60.0:.2f} min | room={step.values['room_c']:.2f} C | "
                    f"m_ref={step.values['m_ref_kg_s']:.4f} kg/s | m_air={step.values['m_air_kg_s']:.4f} kg/s | "
                    f"COP={step.values['cop_system']:.3f}"
                )
                continue

            prev_state = state.copy()
            model.advance_infiltration_disturbance(time_s, dt_s, history[-1])
            model.advance_room_moisture(dt_s, history[-1])
            controller_outputs = control.update(history[-1], plant_config, dt_s)

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return model.residual(x, prev_state, time_s, dt_s)

            unknowns, iters = solve_dynamic_step(residual_fn, unknowns, step_sim_cfg, jacobian_executor)
            step = model.post_process(unknowns, time_s)
            step.values["newton_iterations"] = float(iters)
            step.values.update(controller_outputs)
            history.append(step.values)
            state = step.state_vector
            model.advance_receiver_inlet_flow(step.values.get("compressor_map_flow_kg_s", step.values["m_ref_kg_s"]), dt_s)
            if idx % max(progress_interval, 1) == 0 or idx == len(times) - 1:
                _log(
                    f"[run] t={time_s / 60.0:.2f} min | room={step.values['room_c']:.2f} C | "
                    f"m_ref={step.values['m_ref_kg_s']:.4f} kg/s | m_air={step.values['m_air_kg_s']:.4f} kg/s | "
                    f"COP={step.values['cop_system']:.3f} | newton={iters}"
                )
    finally:
        if jacobian_executor is not None:
            jacobian_executor.shutdown()

    return history


def _run_air_cycle_simulation(plant_config: dict[str, Any]) -> list[dict[str, float]]:
    sim_cfg = plant_config["simulation"]
    model = CascadeSystemModel(plant_config)
    progress_interval = int(sim_cfg.get("progress_interval_steps", 20))

    dt_s = sim_cfg["dt_s"]
    times = np.arange(sim_cfg["t_start_s"], sim_cfg["t_end_s"] + dt_s, dt_s)
    unknowns = air_cycle_initial_vector(plant_config)

    configure_disturbances(plant_config)
    model.reset_room_moisture(float(unknowns[0]))
    model.reset_infiltration_disturbance(
        float(unknowns[0]),
        float(plant_config.get("boundary_conditions", {}).get("dock_initial_c", unknowns[0])),
    )
    state = np.array([unknowns[0], unknowns[1], model.current_room_humidity_ratio()], dtype=float)
    history: list[dict[str, float]] = []
    control: ControlSystem | None = None
    jacobian_workers = max(1, int(sim_cfg.get("jacobian_workers", 1)))
    target_state_len = int(unknowns.size)
    transient_reg_cfg = _air_cycle_transient_regenerator_config(plant_config)
    step_sim_cfg = {
        **sim_cfg,
        "state_lower_bounds": _expanded_air_cycle_solver_vector(
            plant_config,
            sim_cfg.get("air_cycle_state_lower_bounds", sim_cfg.get("state_lower_bounds")),
            [-83.15, -20.0, -100.0, -100.0, -100.0],
            target_state_len,
            float(transient_reg_cfg.get("solid_lower_c", -200.0)) if transient_reg_cfg is not None else -200.0,
        ),
        "state_upper_bounds": _expanded_air_cycle_solver_vector(
            plant_config,
            sim_cfg.get("air_cycle_state_upper_bounds", sim_cfg.get("state_upper_bounds")),
            [46.85, 120.0, 150.0, 150.0, 150.0],
            target_state_len,
            float(transient_reg_cfg.get("solid_upper_c", 150.0)) if transient_reg_cfg is not None else 150.0,
        ),
        "state_step_scales": _expanded_air_cycle_solver_vector(
            plant_config,
            sim_cfg.get("air_cycle_state_step_scales", sim_cfg.get("state_step_scales")),
            [10.0, 10.0, 10.0, 10.0, 10.0],
            target_state_len,
            float(transient_reg_cfg.get("solid_step_scale_k", 10.0)) if transient_reg_cfg is not None else 10.0,
        ),
        "residual_scales": _expanded_air_cycle_solver_vector(
            plant_config,
            sim_cfg.get("air_cycle_residual_scales", sim_cfg.get("residual_scales")),
            [1.0, 1.0, 1.0e5, 1.0e5, 1.0e5],
            target_state_len,
            float(transient_reg_cfg.get("solid_residual_scale_k", 1.0)) if transient_reg_cfg is not None else 1.0,
            transient_base_override=[1.0, 1.0, 1.0, 1.0, 1.0] if transient_reg_cfg is not None else None,
        ),
    }

    _log(f"[run:air_cycle] steps={len(times) - 1} dt={dt_s:.1f}s interval={progress_interval}")
    jacobian_executor = ThreadPoolExecutor(max_workers=jacobian_workers) if jacobian_workers > 1 else None
    try:
        for idx, time_s in enumerate(times):
            if idx == 0:
                step = model.air_cycle_post_process(unknowns, time_s)
                step.values["startup_initialization_enabled"] = 0.0
                step.values["startup_iterations"] = 0.0
                step.values["startup_newton_iterations"] = 0.0
                step.values["startup_cost"] = 0.0
                step.values["newton_iterations"] = 0.0
                history.append(step.values)
                state = step.state_vector
                control = _filtered_control_system(plant_config, step.values)
                _log(
                    f"[startup:air_cycle] t={time_s / 60.0:.2f} min | room={step.values['room_c']:.2f} C | "
                    f"water={step.values['water_loop_c']:.2f} C | m_air={step.values['m_air_kg_s']:.4f} kg/s | "
                    f"COP={step.values['cop_system']:.3f}"
                )
                continue

            prev_state = state.copy()
            model.advance_infiltration_disturbance(time_s, dt_s, history[-1])
            model.advance_room_moisture(dt_s, history[-1])
            controller_outputs = control.update(history[-1], plant_config, dt_s) if control is not None else {}

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return model.air_cycle_residual(x, prev_state, time_s, dt_s)

            unknowns, iters = solve_dynamic_step(residual_fn, unknowns, step_sim_cfg, jacobian_executor)
            step = model.air_cycle_post_process(unknowns, time_s)
            step.values["newton_iterations"] = float(iters)
            step.values.update(controller_outputs)
            history.append(step.values)
            state = step.state_vector
            if idx % max(progress_interval, 1) == 0 or idx == len(times) - 1:
                _log(
                    f"[run:air_cycle] t={time_s / 60.0:.2f} min | room={step.values['room_c']:.2f} C | "
                    f"water={step.values['water_loop_c']:.2f} C | m_air={step.values['m_air_kg_s']:.4f} kg/s | "
                    f"COP={step.values['cop_system']:.3f} | newton={iters}"
                )
    finally:
        if jacobian_executor is not None:
            jacobian_executor.shutdown()

    return history


def _run_vcc_simulation(plant_config: dict[str, Any]) -> list[dict[str, float]]:
    sim_cfg = plant_config["simulation"]
    model = CascadeSystemModel(plant_config)
    progress_interval = int(sim_cfg.get("progress_interval_steps", 20))

    dt_s = sim_cfg["dt_s"]
    times = np.arange(sim_cfg["t_start_s"], sim_cfg["t_end_s"] + dt_s, dt_s)
    unknowns = vcc_initial_vector(plant_config)
    startup = initialize_vcc_standalone_design(plant_config, model, unknowns)
    unknowns = startup.unknowns
    startup_snapshot = startup.snapshot

    state_values = [unknowns[0]]
    if model.high_pressure_receiver_enabled():
        state_values.append(float(unknowns[5]))
    if model.lpr_subcooling_control_enabled():
        subcooling_idx = 5 + (1 if model.high_pressure_receiver_enabled() else 0)
        state_values.append(float(unknowns[subcooling_idx]))
    if model.lpr_inventory_enabled():
        lpr_idx = 5 + (1 if model.high_pressure_receiver_enabled() else 0) + (1 if model.lpr_subcooling_control_enabled() else 0)
        state_values.extend([float(unknowns[lpr_idx]), float(unknowns[lpr_idx + 1])])
    state = np.array(state_values, dtype=float)
    history: list[dict[str, float]] = []
    control: ControlSystem | None = None
    jacobian_workers = max(1, int(sim_cfg.get("jacobian_workers", 1)))
    receiver_cfg = _receiver_config(plant_config)
    state_lower_bounds = [-3.15, -73.15, 0.0, 1.0e-6, 1.0e-6]
    state_upper_bounds = [86.85, 46.85, 90.0, 5.0, 5.0]
    state_step_scales = [10.0, 5.0, 10.0, 0.01, 0.01]
    residual_scales = [1.0, 1.0e5, 0.1, 0.1, 0.1]
    if model.high_pressure_receiver_enabled():
        state_lower_bounds.append(float(receiver_cfg.get("mass_min_kg", 0.0)))
        state_upper_bounds.append(float(receiver_cfg.get("mass_max_kg", receiver_cfg.get("maximum_mass_kg", 100.0))))
        state_step_scales.append(float(receiver_cfg.get("mass_step_scale_kg", max(abs(float(unknowns[5])), 0.1))))
        residual_scales.append(float(receiver_cfg.get("mass_residual_scale_kg", 0.1)))
    if model.lpr_subcooling_control_enabled():
        lpr_cfg = _low_pressure_receiver_config(plant_config)
        subcooling_idx = 5 + (1 if model.high_pressure_receiver_enabled() else 0)
        state_lower_bounds.append(float(lpr_cfg.get("subcooling_min_k", 0.0)))
        state_upper_bounds.append(float(lpr_cfg.get("subcooling_max_k", 30.0)))
        state_step_scales.append(float(lpr_cfg.get("subcooling_step_scale_k", max(abs(float(unknowns[subcooling_idx])), 1.0))))
        residual_scales.append(float(lpr_cfg.get("subcooling_residual_scale_k", 1.0)))
    _append_lpr_inventory_solver_entries(
        plant_config,
        state_lower_bounds,
        state_upper_bounds,
        state_step_scales,
        residual_scales,
        unknowns,
        volume_residual_index=4,
    )
    step_sim_cfg = {
        **sim_cfg,
        "state_lower_bounds": state_lower_bounds,
        "state_upper_bounds": state_upper_bounds,
        "state_step_scales": state_step_scales,
        "residual_scales": residual_scales,
    }

    _log(f"[run:vcc] steps={len(times) - 1} dt={dt_s:.1f}s interval={progress_interval}")
    jacobian_executor = ThreadPoolExecutor(max_workers=jacobian_workers) if jacobian_workers > 1 else None
    try:
        for idx, time_s in enumerate(times):
            if idx == 0:
                step = model.vcc_post_process(unknowns, time_s)
                step.values["startup_initialization_enabled"] = float(sim_cfg.get("startup_initialization", {}).get("enabled", False))
                step.values["startup_iterations"] = float(startup.iterations)
                step.values["startup_newton_iterations"] = float(startup.iterations)
                step.values["startup_cost"] = startup.cost
                for key, value in startup_snapshot.items():
                    if key.startswith("startup_solved_"):
                        step.values[key] = value
                step.values["newton_iterations"] = 0.0
                history.append(step.values)
                state = step.state_vector
                model.reset_receiver_inlet_flow(step.values.get("compressor_map_flow_kg_s", step.values["m_ref_kg_s"]))
                control = _filtered_control_system(plant_config, step.values)
                _log(
                    f"[startup:vcc] t={time_s / 60.0:.2f} min | sink={step.values['sink_c']:.2f} C | "
                    f"tevap={step.values['tevap_c']:.2f} C | m_ref={step.values['m_ref_kg_s']:.4f} kg/s | "
                    f"COP={step.values['cop_system']:.3f}"
                )
                continue

            prev_state = state.copy()
            controller_outputs = control.update(history[-1], plant_config, dt_s) if control is not None else {}

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return model.vcc_residual(x, prev_state, time_s, dt_s)

            unknowns, iters = solve_dynamic_step(residual_fn, unknowns, step_sim_cfg, jacobian_executor)
            step = model.vcc_post_process(unknowns, time_s)
            step.values["newton_iterations"] = float(iters)
            step.values.update(controller_outputs)
            history.append(step.values)
            state = step.state_vector
            model.advance_receiver_inlet_flow(step.values.get("compressor_map_flow_kg_s", step.values["m_ref_kg_s"]), dt_s)
            if idx % max(progress_interval, 1) == 0 or idx == len(times) - 1:
                _log(
                    f"[run:vcc] t={time_s / 60.0:.2f} min | sink={step.values['sink_c']:.2f} C | "
                    f"tevap={step.values['tevap_c']:.2f} C | m_ref={step.values['m_ref_kg_s']:.4f} kg/s | "
                    f"COP={step.values['cop_system']:.3f} | newton={iters}"
                )
    finally:
        if jacobian_executor is not None:
            jacobian_executor.shutdown()

    return history


def save_plot(history: list[dict[str, float]], plot_file: str | Path) -> None:
    out_path = Path(plot_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_min = np.array([row["time_s"] for row in history]) / 60.0
    def temperature_spec(kelvin_key: str, celsius_key: str, label: str) -> tuple[str, str, str]:
        if any(kelvin_key in row for row in history):
            return (kelvin_key, label, "Temperature [K]")
        return (celsius_key, label, "Temperature [C]")

    series_specs = [
        temperature_spec("refrigerating_temperature_k", "refrigerating_temperature_c", "Refrigerating"),
        temperature_spec("room_k", "room_c", "Room"),
        temperature_spec("dock_k", "dock_c", "Loading dock"),
        temperature_spec("water_loop_k", "water_loop_c", "Water loop"),
        temperature_spec("sink_k", "sink_c", "Sink"),
        temperature_spec("tevap_k", "tevap_c", "Evaporating"),
        temperature_spec("tcond_k", "tcond_c", "Condensing"),
        temperature_spec("t3_k", "t3_c", "Air after cascade exchanger"),
        temperature_spec("t5_k", "t5_c", "Air entering room"),
        ("m_ref_kg_s", "Refrigerant mass flow", "Mass Flow [kg/s]"),
        ("m_air_kg_s", "Air mass flow", "Mass Flow [kg/s]"),
        ("cop_system", "COP", "COP"),
    ]
    plotted_series: list[tuple[str, str, str, np.ndarray]] = []
    for key, label, ylabel in series_specs:
        if not any(key in row for row in history):
            continue
        values = np.array([float(row.get(key, np.nan)) for row in history], dtype=float)
        if np.all(np.isnan(values)):
            continue
        plotted_series.append((key, label, ylabel, values))

    if not plotted_series:
        plotted_series.append(("time_s", "Time", "Time [min]", t_min))

    fig, axes = plt.subplots(len(plotted_series), 1, figsize=(10, max(4, 2.4 * len(plotted_series))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.ticklabel_format(axis="y", useOffset=False)

    for ax, (_, label, ylabel, values) in zip(axes, plotted_series):
        ax.plot(t_min, values, label=label)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [min]")

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
