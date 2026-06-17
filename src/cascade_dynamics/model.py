from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compressor_map import (
    ammonia_compressor_eta_is,
    ammonia_compressor_map,
    head_from_mass_flow,
    mass_flow_from_actual_head,
    screw_compressor_pressure_ratio_map,
    volumetric_flow_from_head,
)
from .components import positive_lmtd, single_stream_scaled_ua, turbine_actual_enthalpy
from .fluids import fluid_property, h_refrigerant_liquid, p_sat, props_si
from .humid_air import (
    humid_air_state,
    saturation_humidity_ratio,
    saturated_room_humidity_ratio,
    state_at_enthalpy,
    state_at_entropy,
)
from .infiltration import (
    TianInfiltrationState,
    advance_tian_infiltration,
    air_density,
    apply_infiltration_load_application,
    humidity_ratio_from_config,
    zero_infiltration_result,
)


KELVIN_OFFSET = 273.15
CP_DOCK_AIR = 1005.0
H_FG_WATER = 2.501e6
H_SUBLIMATION_ICE = 2.834e6
MAP_POWER_PRESSURE_LIFT_MODES = {
    "map_power_backcalculate",
    "map_power_lift",
    "power_backcalculate",
}
MAP_COOLING_CAPACITY_BALANCE_MODES = {
    "map_cooling_capacity",
    "map_capacity",
    "map_capacity_direct",
}
DYNAMIC_UA_DEFAULT_EXPONENTS = {
    "regenerator": 0.8,
    "cascade": 0.8,
    "condenser": 0.8,
    "dock_evaporator": 0.6,
    "water_loop_air_cooler": 0.8,
}


def compressor_uses_map_power_pressure_lift(compressor_cfg: dict) -> bool:
    return str(compressor_cfg.get("pressure_lift_model", "")).lower() in MAP_POWER_PRESSURE_LIFT_MODES


def compressor_uses_map_cooling_capacity_balance(compressor_cfg: dict) -> bool:
    return str(compressor_cfg.get("capacity_balance_model", "")).lower() in MAP_COOLING_CAPACITY_BALANCE_MODES


def compressor_discharge_pressure_from_power(
    p_suction_pa: float,
    h_suction_j_kg: float,
    s_suction_j_kg_k: float,
    mass_flow_kg_s: float,
    power_w: float,
    eta_is: float,
    fluid: str,
    pressure_ratio_min: float = 1.000001,
    pressure_ratio_max: float | None = None,
) -> float:
    target_h_is = h_suction_j_kg + max(float(power_w), 0.0) * max(float(eta_is), 1.0e-6) / max(float(mass_flow_kg_s), 1.0e-9)
    lo = max(float(p_suction_pa) * max(float(pressure_ratio_min), 1.000001), float(p_suction_pa) * 1.000001)
    if pressure_ratio_max is None:
        hi = 0.999 * fluid_property("Pcrit", fluid)
    else:
        hi = float(p_suction_pa) * max(float(pressure_ratio_max), float(pressure_ratio_min) * 1.001)
    hi = max(hi, lo * 1.001)

    def residual(pressure_pa: float) -> float:
        h_is = props_si("H", "P", pressure_pa, "S", s_suction_j_kg_k, fluid)
        return h_is - target_h_is

    f_lo = residual(lo)
    if f_lo >= 0.0:
        return lo

    f_hi = residual(hi)
    if f_hi <= 0.0:
        return hi

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) <= 1.0e-4 or abs(hi - lo) <= 1.0e-3:
            return mid
        if f_mid >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def compressor_eta_is_from_map_power(
    p_discharge_pa: float,
    h_suction_j_kg: float,
    s_suction_j_kg_k: float,
    mass_flow_kg_s: float,
    power_w: float,
    fluid: str,
) -> float:
    h_is = props_si("H", "P", p_discharge_pa, "S", s_suction_j_kg_k, fluid)
    return max(float(mass_flow_kg_s), 1.0e-9) * max(h_is - h_suction_j_kg, 0.0) / max(float(power_w), 1.0e-9)


def compressor_volumetric_efficiency_clearance(
    pressure_ratio: float,
    clearance_factor: float,
    polytropic_exponent: float,
    eta_min: float = 0.3,
    eta_max: float = 1.0,
) -> float:
    eta_v = 1.0 + float(clearance_factor) - float(clearance_factor) * max(float(pressure_ratio), 1.0) ** (
        1.0 / max(float(polytropic_exponent), 1.0e-9)
    )
    return float(np.clip(eta_v, eta_min, eta_max))


def compressor_mass_flow_positive_displacement(
    suction_density_kg_m3: float,
    speed_rpm: float,
    displacement_m3_per_rev: float,
    volumetric_efficiency: float,
) -> float:
    return (
        max(float(suction_density_kg_m3), 0.0)
        * max(float(displacement_m3_per_rev), 0.0)
        * max(float(speed_rpm), 0.0)
        / 60.0
        * max(float(volumetric_efficiency), 0.0)
    )


def expansion_valve_flow_factor(
    p_upstream_pa: float,
    p_downstream_pa: float,
    inlet_density_kg_m3: float,
) -> float:
    return float(np.sqrt(2.0 * max(float(inlet_density_kg_m3), 0.0) * max(float(p_upstream_pa) - float(p_downstream_pa), 0.0)))


def expansion_valve_flow_coefficient(valve_cfg: dict) -> float:
    return float(
        valve_cfg.get(
            "flow_coefficient_m2",
            valve_cfg.get("flow_coefficient_kg_s_sqrt_pa_density", valve_cfg.get("flow_coefficient_kg_s_pa", 0.0)),
        )
    )


@dataclass
class StepResult:
    values: dict[str, float]
    state_vector: np.ndarray


@dataclass
class RoomMoistureState:
    humidity_ratio: float = 0.0
    relative_humidity: float = 0.0
    saturation_humidity_ratio: float = 0.0
    dry_air_mass_kg: float = 0.0
    deposition_rate_kg_s: float = 0.0
    q_deposition_w: float = 0.0
    cumulative_deposited_water_kg: float = 0.0
    infiltration_dry_air_mass_flow_kg_s: float = 0.0


class CascadeSystemModel:
    def __init__(self, config: dict):
        self.cfg = config
        self.air_fluid = config["fluids"]["air"]
        self.ref_fluid = config["fluids"]["refrigerant"]
        self._infiltration_state = TianInfiltrationState()
        self._current_infiltration = zero_infiltration_result()
        self._room_moisture = RoomMoistureState()
        self._room_moisture_initialized = False
        self._branch_holdup_h: dict[str, float] = {}
        self._branch_holdup_time_s: float | None = None
        self._receiver_inlet_m_dot_kg_s: float | None = None

    @staticmethod
    def _first_positive_config_value(config: dict, keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key not in config:
                continue
            try:
                value = float(config[key])
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return value
        return None

    def refresh_dynamic_ua_nominal_flows(self, unknowns: np.ndarray | None = None) -> None:
        air_cfg = self.cfg["air_cycle"]
        flow_cfg = air_cfg.get("compressor_mass_flow", {})
        air_nominal = self._first_positive_config_value(
            flow_cfg,
            ("nominal_m_dot_kg_s", "design_m_dot_kg_s", "startup_target_m_dot_kg_s"),
        )
        if air_nominal is None:
            if unknowns is not None:
                room_c, _, t3_c, t4_c, t6_c, _, _, _, _, _ = unknowns[:10]
                air = self._evaluate_air_cycle(
                    room_c + KELVIN_OFFSET,
                    t3_c + KELVIN_OFFSET,
                    t4_c + KELVIN_OFFSET,
                    t6_c + KELVIN_OFFSET,
                )
                air_nominal = float(air["m_air"])
            else:
                air_nominal = self._first_positive_config_value(flow_cfg, ("fixed_m_dot_kg_s",))
            if air_nominal is not None and air_nominal > 0.0:
                flow_cfg["nominal_m_dot_kg_s"] = float(air_nominal)

        bc = self.cfg["boundary_conditions"]
        if self._first_positive_config_value(bc, ("sink_nominal_m_dot_kg_s", "sink_design_m_dot_kg_s")) is None:
            sink_nominal = self._first_positive_config_value(bc, ("sink_m_dot_kg_s",))
            if sink_nominal is not None:
                bc["sink_nominal_m_dot_kg_s"] = float(sink_nominal)

        dock_cfg = self.cfg["vcc_cycle"].get("dock_evaporator", {})
        if isinstance(dock_cfg, dict) and self._first_positive_config_value(
            dock_cfg,
            ("nominal_air_m_dot_kg_s", "design_air_m_dot_kg_s"),
        ) is None:
            dock_air_nominal = self._first_positive_config_value(dock_cfg, ("air_m_dot_kg_s",))
            if dock_air_nominal is not None:
                dock_cfg["nominal_air_m_dot_kg_s"] = float(dock_air_nominal)

    def _ua_scaling_options(self, component_name: str, component_cfg: dict | None = None) -> dict:
        global_cfg = self.cfg.get("heat_exchanger_ua_scaling", self.cfg.get("dynamic_ua", {}))
        if not isinstance(global_cfg, dict):
            global_cfg = {}

        options = dict(global_cfg)
        component_overrides = global_cfg.get(component_name, {})
        if isinstance(component_overrides, dict):
            options.update(component_overrides)

        if component_cfg is not None:
            local_cfg = component_cfg.get("ua_scaling", component_cfg.get("dynamic_ua", {}))
            if isinstance(local_cfg, dict):
                options.update(local_cfg)
        return options

    def _scaled_ua_result(
        self,
        component_name: str,
        ua_nominal_w_k: float,
        m_dot_kg_s: float,
        m_dot_nominal_kg_s: float,
        component_cfg: dict | None = None,
    ) -> dict[str, float]:
        options = self._ua_scaling_options(component_name, component_cfg)
        ua_ref_w_k = max(float(ua_nominal_w_k), 0.0)
        nominal_m_dot = float(
            options.get(
                "nominal_m_dot_kg_s",
                options.get("m_dot_nominal_kg_s", m_dot_nominal_kg_s),
            )
        )
        exponent = float(options.get("exponent", DYNAMIC_UA_DEFAULT_EXPONENTS[component_name]))
        flow_ratio = max(float(m_dot_kg_s), 0.0) / nominal_m_dot if nominal_m_dot > 0.0 else 1.0

        if options.get("enabled", True):
            ua_w_k = single_stream_scaled_ua(
                ua_ref_w_k,
                m_dot_kg_s,
                nominal_m_dot,
                exponent,
                min_flow_ratio=float(options.get("min_flow_ratio", 0.0)),
                max_flow_ratio=options.get("max_flow_ratio"),
            )
        else:
            ua_w_k = ua_ref_w_k

        return {
            "ua_w_k": ua_w_k,
            "ua_ref_w_k": ua_ref_w_k,
            "m_dot_kg_s": max(float(m_dot_kg_s), 0.0),
            "m_dot_nominal_kg_s": max(nominal_m_dot, 0.0),
            "flow_ratio": max(flow_ratio, 0.0),
            "exponent": exponent,
        }

    def _air_nominal_m_dot(self, current_m_air_kg_s: float) -> float:
        flow_cfg = self.cfg["air_cycle"].get("compressor_mass_flow", {})
        configured = self._first_positive_config_value(
            flow_cfg,
            ("nominal_m_dot_kg_s", "design_m_dot_kg_s", "startup_target_m_dot_kg_s", "fixed_m_dot_kg_s"),
        )
        return float(configured if configured is not None else current_m_air_kg_s)

    def _sink_nominal_m_dot(self, current_sink_m_dot_kg_s: float) -> float:
        bc = self.cfg["boundary_conditions"]
        configured = self._first_positive_config_value(
            bc,
            ("sink_nominal_m_dot_kg_s", "sink_design_m_dot_kg_s", "nominal_sink_m_dot_kg_s", "design_sink_m_dot_kg_s"),
        )
        return float(configured if configured is not None else current_sink_m_dot_kg_s)

    def _heat_exchanger_uas(self, air: dict[str, float]) -> dict[str, dict[str, float]]:
        air_nominal = self._air_nominal_m_dot(air["m_air"])
        sink_m_dot = float(self.cfg["boundary_conditions"]["sink_m_dot_kg_s"])
        return {
            "regenerator": self._scaled_ua_result(
                "regenerator",
                self.cfg["air_cycle"]["regenerator_ua_w_k"],
                air["m_air"],
                air_nominal,
            ),
            "cascade": self._scaled_ua_result(
                "cascade",
                self.cfg["vcc_cycle"]["cascade_ua_w_k"],
                air["m_air"],
                air_nominal,
            ),
            "condenser": self._scaled_ua_result(
                "condenser",
                self.cfg["vcc_cycle"]["condenser_ua_w_k"],
                sink_m_dot,
                self._sink_nominal_m_dot(sink_m_dot),
            ),
        }

    def _water_loop_config(self) -> dict:
        air_cfg = self.cfg.get("air_cycle", {})
        vcc_cfg = self.cfg.get("vcc_cycle", {})
        bc = self.cfg.get("boundary_conditions", {})
        caps = self.cfg.get("thermal_masses", {})

        cfg = dict(self.cfg.get("water_loop", {}))
        local_cfg = air_cfg.get("water_loop", {})
        if isinstance(local_cfg, dict):
            cfg.update(local_cfg)

        cfg.setdefault("ambient_c", bc.get("ambient_c", 30.0))
        cfg.setdefault("cascade_ua_w_k", air_cfg.get("cascade_ua_w_k", vcc_cfg.get("cascade_ua_w_k", 0.0)))
        cfg.setdefault("air_cooler_ua_w_k", cfg.get("cooler_ua_w_k", vcc_cfg.get("condenser_ua_w_k", 0.0)))
        cfg.setdefault("capacitance_j_k", caps.get("water_loop_capacitance_j_k", caps.get("sink_capacitance_j_k", 500000.0)))
        cfg.setdefault("initial_c", self.cfg.get("initial_guess", {}).get("water_loop_c", self.cfg.get("initial_guess", {}).get("sink_c", cfg["ambient_c"])))
        return cfg

    def _water_loop_air_cooler_ua_result(self) -> dict[str, float]:
        loop_cfg = self._water_loop_config()
        cooler_cfg = loop_cfg.get("air_cooler", {})
        if not isinstance(cooler_cfg, dict):
            cooler_cfg = {}
        ua_ref_w_k = float(cooler_cfg.get("ua_w_k", loop_cfg.get("air_cooler_ua_w_k", 0.0)))
        m_dot = self._first_positive_config_value(
            cooler_cfg,
            ("air_m_dot_kg_s", "fan_air_m_dot_kg_s", "m_dot_kg_s"),
        )
        if m_dot is None:
            m_dot = self._first_positive_config_value(
                loop_cfg,
                ("air_cooler_air_m_dot_kg_s", "cooler_air_m_dot_kg_s", "air_m_dot_kg_s"),
            )
        if m_dot is None:
            m_dot = 1.0
        nominal_m_dot = self._first_positive_config_value(
            cooler_cfg,
            ("nominal_air_m_dot_kg_s", "design_air_m_dot_kg_s", "nominal_m_dot_kg_s"),
        )
        if nominal_m_dot is None:
            nominal_m_dot = self._first_positive_config_value(
                loop_cfg,
                (
                    "air_cooler_nominal_air_m_dot_kg_s",
                    "air_cooler_design_air_m_dot_kg_s",
                    "nominal_air_m_dot_kg_s",
                ),
            )
        if nominal_m_dot is None:
            nominal_m_dot = m_dot
        component_cfg = {**loop_cfg, **cooler_cfg}
        return self._scaled_ua_result("water_loop_air_cooler", ua_ref_w_k, m_dot, nominal_m_dot, component_cfg)

    def _standalone_air_cycle_uas(self, air: dict[str, float]) -> dict[str, dict[str, float]]:
        loop_cfg = self._water_loop_config()
        cascade_cfg = loop_cfg.get("cascade_exchanger", loop_cfg.get("cascade", {}))
        if not isinstance(cascade_cfg, dict):
            cascade_cfg = {}
        air_nominal = self._air_nominal_m_dot(air["m_air"])
        return {
            "regenerator": self._scaled_ua_result(
                "regenerator",
                self.cfg["air_cycle"]["regenerator_ua_w_k"],
                air["m_air"],
                air_nominal,
            ),
            "cascade": self._scaled_ua_result(
                "cascade",
                float(cascade_cfg.get("ua_w_k", loop_cfg.get("cascade_ua_w_k", 0.0))),
                air["m_air"],
                air_nominal,
                cascade_cfg,
            ),
            "water_loop_air_cooler": self._water_loop_air_cooler_ua_result(),
        }

    def _evaluate_water_loop_air_cooler(self, water_c: float) -> dict[str, float]:
        loop_cfg = self._water_loop_config()
        ua = self._water_loop_air_cooler_ua_result()
        ambient_c = float(loop_cfg.get("ambient_c", self.cfg.get("boundary_conditions", {}).get("ambient_c", 30.0)))
        delta_t_k = float(water_c) - ambient_c
        q_w = ua["ua_w_k"] * delta_t_k
        return {
            "q_w": q_w,
            "ua_w_k": ua["ua_w_k"],
            "ua_ref_w_k": ua["ua_ref_w_k"],
            "flow_ratio": ua["flow_ratio"],
            "ua_flow_exponent": ua["exponent"],
            "ambient_c": ambient_c,
            "delta_t_k": delta_t_k,
        }

    def _vcc_standalone_loads_w(self, time_s: float) -> dict[str, float]:
        bc = self.cfg.get("boundary_conditions", {})
        vcc_cfg = self.cfg.get("vcc_cycle", {})
        standalone_cfg = vcc_cfg.get("standalone", vcc_cfg.get("validation", {}))
        if not isinstance(standalone_cfg, dict):
            standalone_cfg = {}
        loads_cfg = standalone_cfg.get("evaporator_loads", vcc_cfg.get("standalone_evaporator_loads", {}))
        if not isinstance(loads_cfg, dict):
            loads_cfg = {}
        after_step = float(time_s) >= float(loads_cfg.get("load_step_time_s", bc.get("load_step_time_s", float("inf"))))

        def scheduled_value(prefix: str, fallback_before: float, fallback_after: float | None = None) -> float:
            fixed_key = f"{prefix}_w"
            before_key = f"{prefix}_before_w"
            after_key = f"{prefix}_after_w"
            if fixed_key in loads_cfg:
                return float(loads_cfg[fixed_key])
            before = float(loads_cfg.get(before_key, fallback_before))
            after = float(loads_cfg.get(after_key, fallback_after if fallback_after is not None else before))
            return after if after_step else before

        total_is_explicit = "total_w" in loads_cfg or "total_before_w" in loads_cfg or "total_after_w" in loads_cfg
        branch_is_explicit = any(
            key in loads_cfg
            for key in (
                "cascade_w",
                "cascade_before_w",
                "cascade_after_w",
                "dock_w",
                "dock_before_w",
                "dock_after_w",
            )
        )
        if total_is_explicit and not branch_is_explicit:
            total = scheduled_value(
                "total",
                float(bc.get("vcc_evaporator_load_before_w", bc.get("load_before_w", 0.0))),
                float(bc.get("vcc_evaporator_load_after_w", bc.get("load_after_w", bc.get("load_before_w", 0.0)))),
            )
            return {"cascade_w": max(total, 0.0), "dock_w": 0.0, "total_w": max(total, 0.0)}

        cascade = scheduled_value(
            "cascade",
            float(bc.get("vcc_cascade_load_before_w", bc.get("load_before_w", 0.0))),
            float(bc.get("vcc_cascade_load_after_w", bc.get("load_after_w", bc.get("load_before_w", 0.0)))),
        )
        dock = scheduled_value(
            "dock",
            float(bc.get("vcc_dock_load_before_w", bc.get("dock_load_before_w", 0.0))),
            float(bc.get("vcc_dock_load_after_w", bc.get("dock_load_after_w", bc.get("dock_load_before_w", 0.0)))),
        )
        cascade = max(float(cascade), 0.0)
        dock = max(float(dock), 0.0)
        return {"cascade_w": cascade, "dock_w": dock, "total_w": cascade + dock}

    def _vcc_condenser_ua_result(self) -> dict[str, float]:
        sink_m_dot = float(self.cfg["boundary_conditions"].get("sink_m_dot_kg_s", 0.0))
        return self._scaled_ua_result(
            "condenser",
            self.cfg["vcc_cycle"]["condenser_ua_w_k"],
            sink_m_dot,
            self._sink_nominal_m_dot(sink_m_dot),
        )

    def _receiver_config(self) -> dict:
        vcc_cfg = self.cfg.get("vcc_cycle", {})
        receiver_cfg = vcc_cfg.get("receiver", vcc_cfg.get("high_pressure_receiver", {}))
        return receiver_cfg if isinstance(receiver_cfg, dict) else {}

    def _vcc_layout(self) -> str:
        vcc_cfg = self.cfg.get("vcc_cycle", {})
        layout = vcc_cfg.get("layout", vcc_cfg.get("configuration", vcc_cfg.get("cycle_layout", "")))
        return str(layout or "").strip().lower()

    def high_pressure_receiver_enabled(self) -> bool:
        layout = self._vcc_layout()
        return bool(self._receiver_config().get("enabled", False)) or layout in {"hpr", "high_pressure_receiver"}

    def _hpr_forces_saturated_liquid(self) -> bool:
        cfg = self._receiver_config()
        return self.high_pressure_receiver_enabled() and bool(cfg.get("force_saturated_liquid_outlet", True))

    def _low_pressure_receiver_config(self) -> dict:
        vcc_cfg = self.cfg.get("vcc_cycle", {})
        receiver_cfg = vcc_cfg.get("low_pressure_receiver", vcc_cfg.get("lpr", {}))
        return receiver_cfg if isinstance(receiver_cfg, dict) else {}

    def low_pressure_receiver_enabled(self) -> bool:
        layout = self._vcc_layout()
        return bool(self._low_pressure_receiver_config().get("enabled", False)) or layout in {"lpr", "low_pressure_receiver"}

    def vcc_evaporators_are_series(self) -> bool:
        vcc_cfg = self.cfg.get("vcc_cycle", {})
        lpr_cfg = self._low_pressure_receiver_config()
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

    def effective_refrigerant_mass_flow(self, m_ref_cascade: float, m_ref_dock: float) -> float:
        if self.vcc_evaporators_are_series():
            return 0.5 * (max(float(m_ref_cascade), 0.0) + max(float(m_ref_dock), 0.0))
        return max(float(m_ref_cascade), 0.0) + max(float(m_ref_dock), 0.0)

    def _compressor_uses_saturated_suction(self) -> bool:
        cfg = self._low_pressure_receiver_config()
        return self.low_pressure_receiver_enabled() and bool(cfg.get("force_saturated_suction", True))

    def lpr_subcooling_control_enabled(self) -> bool:
        cfg = self._low_pressure_receiver_config()
        return self.low_pressure_receiver_enabled() and bool(
            cfg.get("control_subcooling_with_eev", cfg.get("dynamic_subcooling", False))
        )

    def _lpr_subcooling_target_k(self) -> float:
        cfg = self._low_pressure_receiver_config()
        return max(
            float(
                cfg.get(
                    "subcooling_setpoint_k",
                    cfg.get("target_subcooling_k", self.cfg.get("vcc_cycle", {}).get("subcooling_k", 0.0)),
                )
            ),
            0.0,
        )

    def _lpr_subcooling_unknown(self, unknowns: np.ndarray, base_size: int, default: float | None = None) -> float:
        if not self.lpr_subcooling_control_enabled():
            return max(float(self.cfg.get("vcc_cycle", {}).get("subcooling_k", 0.0)), 0.0)
        idx = base_size + (1 if self.high_pressure_receiver_enabled() else 0)
        if len(unknowns) > idx:
            return max(float(unknowns[idx]), 0.0)
        if default is not None:
            return max(float(default), 0.0)
        return self._lpr_subcooling_target_k()

    def _receiver_extra_residual_count(self) -> int:
        return (1 if self.high_pressure_receiver_enabled() else 0) + (
            1 if self.lpr_subcooling_control_enabled() else 0
        )

    def _receiver_unknown_mass(self, unknowns: np.ndarray, default: float = 0.0) -> float:
        if self.high_pressure_receiver_enabled() and len(unknowns) > 10:
            return max(float(unknowns[10]), 0.0)
        return max(float(default), 0.0)

    def _vcc_receiver_unknown_mass(self, unknowns: np.ndarray, default: float = 0.0) -> float:
        if self.high_pressure_receiver_enabled() and len(unknowns) > 5:
            return max(float(unknowns[5]), 0.0)
        return max(float(default), 0.0)

    def _receiver_inlet_flow_kg_s(self, fallback_kg_s: float) -> float:
        if not self.high_pressure_receiver_enabled():
            return 0.0
        if self._receiver_inlet_m_dot_kg_s is None:
            return max(float(fallback_kg_s), 0.0)
        return max(float(self._receiver_inlet_m_dot_kg_s), 0.0)

    def reset_receiver_inlet_flow(self, m_dot_kg_s: float) -> None:
        if self.high_pressure_receiver_enabled():
            self._receiver_inlet_m_dot_kg_s = max(float(m_dot_kg_s), 0.0)

    def advance_receiver_inlet_flow(self, m_dot_kg_s: float, dt_s: float) -> None:
        if not self.high_pressure_receiver_enabled():
            return
        target = max(float(m_dot_kg_s), 0.0)
        if self._receiver_inlet_m_dot_kg_s is None:
            self._receiver_inlet_m_dot_kg_s = target
            return
        tau_s = float(
            self._receiver_config().get(
                "inlet_time_constant_s",
                self._receiver_config().get("condenser_outlet_time_constant_s", 5.0),
            )
        )
        if tau_s <= 0.0:
            self._receiver_inlet_m_dot_kg_s = target
            return
        alpha = min(max(float(dt_s) / tau_s, 0.0), 1.0)
        self._receiver_inlet_m_dot_kg_s += alpha * (target - self._receiver_inlet_m_dot_kg_s)

    def _evaluate_high_pressure_receiver(
        self,
        receiver_mass_kg: float | None,
        p_cond_pa: float,
        tcond_k: float,
        liquid_enthalpy_j_kg: float,
    ) -> dict[str, float]:
        cfg = self._receiver_config()
        valve_inlet_density = props_si("D", "P", p_cond_pa, "H", liquid_enthalpy_j_kg, self.ref_fluid)
        disabled = {
            "enabled": 0.0,
            "mass_kg": 0.0,
            "volume_m3": 0.0,
            "liquid_mass_kg": 0.0,
            "vapor_mass_kg": 0.0,
            "liquid_volume_m3": 0.0,
            "liquid_fill_fraction": 1.0,
            "liquid_fill_fraction_raw": 1.0,
            "overfill_mass_kg": 0.0,
            "liquid_feed_fraction": 1.0,
            "starvation_flow_multiplier": 1.0,
            "outlet_enthalpy_j_kg": liquid_enthalpy_j_kg,
            "outlet_density_kg_m3": valve_inlet_density,
            "inlet_m_dot_kg_s": 0.0,
        }
        if not self.high_pressure_receiver_enabled():
            return disabled

        mass_kg = max(float(receiver_mass_kg or 0.0), 0.0)
        volume_m3 = max(float(cfg.get("volume_m3", cfg.get("internal_volume_m3", 0.0))), 0.0)
        rho_l = props_si("D", "T", tcond_k, "Q", 0.0, self.ref_fluid)
        rho_v = props_si("D", "T", tcond_k, "Q", 1.0, self.ref_fluid)
        h_v = props_si("H", "T", tcond_k, "Q", 1.0, self.ref_fluid)

        if volume_m3 <= 0.0:
            liquid_mass_kg = mass_kg
            vapor_mass_kg = 0.0
            liquid_volume_m3 = mass_kg / max(rho_l, 1.0e-9)
            liquid_fill_raw = 1.0
            overfill_mass_kg = 0.0
        else:
            liquid_volume_raw = (mass_kg - rho_v * volume_m3) / max(rho_l - rho_v, 1.0e-9)
            liquid_volume_m3 = float(np.clip(liquid_volume_raw, 0.0, volume_m3))
            vapor_volume_m3 = max(volume_m3 - liquid_volume_m3, 0.0)
            phase_mass_kg = rho_l * liquid_volume_m3 + rho_v * vapor_volume_m3
            overfill_mass_kg = max(mass_kg - phase_mass_kg, 0.0)
            liquid_mass_kg = rho_l * liquid_volume_m3 + overfill_mass_kg
            vapor_mass_kg = rho_v * vapor_volume_m3
            liquid_fill_raw = liquid_volume_raw / volume_m3

        min_liquid_mass = cfg.get("minimum_liquid_mass_kg")
        if min_liquid_mass is None:
            min_liquid_fraction = float(cfg.get("minimum_liquid_fill_fraction", 0.02))
            min_liquid_mass = min_liquid_fraction * max(volume_m3, 0.0) * max(rho_l, 0.0)
        smoothing_mass = cfg.get("starvation_smoothing_mass_kg")
        if smoothing_mass is None:
            smoothing_fraction = float(cfg.get("starvation_smoothing_fill_fraction", 0.05))
            smoothing_mass = smoothing_fraction * max(volume_m3, 0.0) * max(rho_l, 0.0)
        smoothing_mass = max(float(smoothing_mass), 1.0e-9)
        feed_fraction = float(np.clip((liquid_mass_kg - float(min_liquid_mass)) / smoothing_mass, 0.0, 1.0))

        if bool(cfg.get("model_vapor_feed_when_empty", True)):
            outlet_enthalpy = feed_fraction * liquid_enthalpy_j_kg + (1.0 - feed_fraction) * h_v
            outlet_density = feed_fraction * rho_l + (1.0 - feed_fraction) * rho_v
        else:
            outlet_enthalpy = liquid_enthalpy_j_kg
            outlet_density = valve_inlet_density

        min_flow_fraction = float(cfg.get("minimum_flow_fraction_when_empty", 0.02))
        if bool(cfg.get("limit_valve_flow_when_starved", True)):
            flow_multiplier = min_flow_fraction + (1.0 - min_flow_fraction) * feed_fraction
        else:
            flow_multiplier = 1.0

        return {
            "enabled": 1.0,
            "mass_kg": mass_kg,
            "volume_m3": volume_m3,
            "liquid_mass_kg": liquid_mass_kg,
            "vapor_mass_kg": vapor_mass_kg,
            "liquid_volume_m3": liquid_volume_m3,
            "liquid_fill_fraction": float(np.clip(liquid_fill_raw, 0.0, 1.0)),
            "liquid_fill_fraction_raw": liquid_fill_raw,
            "overfill_mass_kg": overfill_mass_kg,
            "liquid_feed_fraction": feed_fraction,
            "starvation_flow_multiplier": flow_multiplier,
            "outlet_enthalpy_j_kg": outlet_enthalpy,
            "outlet_density_kg_m3": max(outlet_density, 1.0e-9),
            "inlet_m_dot_kg_s": self._receiver_inlet_flow_kg_s(0.0),
        }

    def _receiver_mass_balance_residual(
        self,
        receiver_mass_kg: float,
        previous_receiver_mass_kg: float,
        ref: dict[str, float],
        dt_s: float,
    ) -> float:
        inlet_m_dot = self._receiver_inlet_flow_kg_s(ref["m_ref_compressor"])
        return (
            float(receiver_mass_kg)
            - float(previous_receiver_mass_kg)
            - float(dt_s) * (inlet_m_dot - float(ref["m_ref_valve"]))
        )

    def _lpr_subcooling_balance_residual(
        self,
        subcooling_k: float,
        previous_subcooling_k: float,
        ref: dict[str, float],
        dt_s: float,
    ) -> float:
        cfg = self._low_pressure_receiver_config()
        gain_k_per_kg = float(cfg.get("subcooling_inventory_gain_k_per_kg", 30.0))
        relaxation_s = float(cfg.get("subcooling_relaxation_time_s", 0.0))
        target_k = self._lpr_subcooling_target_k()
        flow_imbalance = float(ref["m_ref_compressor"]) - float(ref["m_ref_valve"])
        residual = float(subcooling_k) - float(previous_subcooling_k) - float(dt_s) * gain_k_per_kg * flow_imbalance
        if relaxation_s > 0.0:
            residual += float(dt_s) * (float(previous_subcooling_k) - target_k) / relaxation_s
        return residual

    def _infiltration_cfg(self) -> dict:
        return self.cfg.get("disturbances", {}).get("infiltration", {})

    def _uses_tian_infiltration(self) -> bool:
        cfg = self._infiltration_cfg()
        mode = str(cfg.get("model", cfg.get("magnitude_mode", ""))).lower()
        return cfg.get("enabled", False) and mode in {"tian", "tian_unsteady", "tian_analytical"}

    def _configured_room_humidity_ratio(self, room_k: float, pressure_pa: float) -> float:
        humidity_cfg = self.cfg["air_cycle"].get("humid_air", {})
        x_room = humidity_cfg.get("humidity_ratio")
        if x_room is not None:
            return float(x_room)
        relative_humidity = humidity_cfg.get("room_relative_humidity", humidity_cfg.get("initial_relative_humidity", 1.0))
        return saturated_room_humidity_ratio(room_k, pressure_pa, float(relative_humidity))

    def _room_humidity_ratio(self, room_k: float, pressure_pa: float) -> float:
        if self._room_moisture_initialized:
            return float(self._room_moisture.humidity_ratio)
        return self._configured_room_humidity_ratio(room_k, pressure_pa)

    def _room_air_volume_m3(self) -> float:
        humidity_cfg = self.cfg["air_cycle"].get("humid_air", {})
        for key in ("room_volume_m3", "volume_m3"):
            if key in humidity_cfg:
                return max(float(humidity_cfg[key]), 1.0e-9)

        infiltration_cfg = self._infiltration_cfg()
        for source_key in ("room", "indoor"):
            room_cfg = infiltration_cfg.get(source_key, {})
            if not isinstance(room_cfg, dict):
                continue
            width_m = room_cfg.get("width_m")
            height_m = room_cfg.get("height_m")
            length_m = room_cfg.get("length_m", room_cfg.get("depth_m"))
            if width_m is not None and height_m is not None and length_m is not None:
                return max(float(width_m) * float(height_m) * float(length_m), 1.0e-9)

        return max(float(humidity_cfg.get("default_room_volume_m3", 100.0)), 1.0e-9)

    def _room_dry_air_mass_kg(self, room_k: float, pressure_pa: float, humidity_ratio: float) -> float:
        humidity_cfg = self.cfg["air_cycle"].get("humid_air", {})
        if "room_dry_air_mass_kg" in humidity_cfg:
            return max(float(humidity_cfg["room_dry_air_mass_kg"]), 1.0e-9)
        state = humid_air_state(room_k, pressure_pa, humidity_ratio)
        return max(state.dry_air_density_kg_m3 * self._room_air_volume_m3(), 1.0e-9)

    def _update_room_moisture_properties(self, room_k: float, pressure_pa: float) -> None:
        omega = max(float(self._room_moisture.humidity_ratio), 0.0)
        omega_sat = saturation_humidity_ratio(room_k, pressure_pa)
        self._room_moisture.humidity_ratio = omega
        self._room_moisture.saturation_humidity_ratio = omega_sat
        self._room_moisture.relative_humidity = omega / max(omega_sat, 1.0e-12)
        self._room_moisture.dry_air_mass_kg = self._room_dry_air_mass_kg(room_k, pressure_pa, omega)

    def reset_room_moisture(self, room_c: float) -> None:
        p1 = float(self.cfg["air_cycle"]["p_low_pa"])
        room_k = float(room_c) + KELVIN_OFFSET
        omega = self._configured_room_humidity_ratio(room_k, p1)
        self._room_moisture = RoomMoistureState(humidity_ratio=max(omega, 0.0))
        self._room_moisture_initialized = True
        self._update_room_moisture_properties(room_k, p1)

    def current_room_humidity_ratio(self) -> float:
        return float(self._room_moisture.humidity_ratio)

    def _infiltration_affects_room_moisture(self) -> bool:
        cfg = self._infiltration_cfg()
        if not cfg.get("enabled", False):
            return False
        source = self._infiltration_indoor_source()
        return source in {"room", "cold_room", "refrigerated_space"}

    def advance_room_moisture(self, dt_s: float, previous_values: dict[str, float]) -> None:
        if not self._room_moisture_initialized:
            self.reset_room_moisture(float(previous_values["room_c"]))

        p1 = float(self.cfg["air_cycle"]["p_low_pa"])
        room_k = float(previous_values["room_c"]) + KELVIN_OFFSET
        omega_old = max(float(self._room_moisture.humidity_ratio), 0.0)
        m_da_room = self._room_dry_air_mass_kg(room_k, p1, omega_old)
        m_da_in = 0.0
        omega_outdoor = omega_old

        if self._infiltration_affects_room_moisture():
            infiltration = self._current_infiltration
            q_m3_s = max(float(infiltration.get("q_m3_s", 0.0)), 0.0)
            omega_outdoor = float(infiltration.get("omega_outdoor_kg_kg_da", omega_old))
            rho_outdoor = max(float(infiltration.get("rho_outdoor_kg_m3", 0.0)), 0.0)
            if q_m3_s > 0.0 and rho_outdoor > 0.0:
                m_da_in = q_m3_s * rho_outdoor / max(1.0 + max(omega_outdoor, 0.0), 1.0e-9)

        omega_trial = omega_old
        if dt_s > 0.0 and m_da_in > 0.0:
            omega_trial += float(dt_s) * m_da_in * (omega_outdoor - omega_old) / m_da_room
        omega_trial = max(omega_trial, 0.0)

        omega_sat = saturation_humidity_ratio(room_k, p1)
        deposited_water_kg = max((omega_trial - omega_sat) * m_da_room, 0.0)
        if deposited_water_kg > 0.0:
            omega_new = omega_sat
        else:
            omega_new = omega_trial

        h_phase = float(
            self.cfg["air_cycle"]
            .get("humid_air", {})
            .get("h_deposition_j_kg", H_SUBLIMATION_ICE if room_k < KELVIN_OFFSET else H_FG_WATER)
        )
        deposition_rate = deposited_water_kg / max(float(dt_s), 1.0e-9)
        self._room_moisture.humidity_ratio = omega_new
        self._room_moisture.deposition_rate_kg_s = deposition_rate
        self._room_moisture.q_deposition_w = deposition_rate * h_phase
        self._room_moisture.cumulative_deposited_water_kg += deposited_water_kg
        self._room_moisture.infiltration_dry_air_mass_flow_kg_s = m_da_in
        self._update_room_moisture_properties(room_k, p1)

    def _infiltration_indoor_source(self) -> str:
        return str(self._infiltration_cfg().get("indoor_source", "room")).lower()

    def _infiltration_indoor_temperature_c(self, cfg: dict, values: dict[str, float]) -> float:
        source = self._infiltration_indoor_source()
        if source in {"dock", "loading_dock"}:
            return float(values.get("dock_c", self.cfg["boundary_conditions"]["dock_initial_c"]))
        if source == "fixed":
            return float(cfg.get("indoor_c", cfg.get("T_i_c", values["room_c"])))
        if source in {"room", "cold_room", "refrigerated_space"}:
            return float(values["room_c"])
        raise ValueError(f"Unsupported infiltration indoor_source: {source}")

    def _infiltration_indoor_humidity_ratio(self, indoor_k: float, pressure_pa: float) -> float | None:
        source = self._infiltration_indoor_source()
        cfg = self._infiltration_cfg()
        omega_indoor = humidity_ratio_from_config(cfg, "indoor", indoor_k, pressure_pa)
        if omega_indoor is not None:
            return omega_indoor
        if source in {"dock", "loading_dock"}:
            return humidity_ratio_from_config(cfg, "dock", indoor_k, pressure_pa)
        if source in {"room", "cold_room", "refrigerated_space"}:
            return self._room_humidity_ratio(indoor_k, pressure_pa)
        return None

    def reset_infiltration_disturbance(self, room_c: float, dock_c: float) -> None:
        cfg = self._infiltration_cfg()
        if not self._uses_tian_infiltration():
            self._current_infiltration = zero_infiltration_result()
            return
        p_i = float(cfg.get("P_i", cfg.get("indoor_pressure_pa", self.cfg["air_cycle"]["p_low_pa"])))
        indoor_c = self._infiltration_indoor_temperature_c(cfg, {"room_c": room_c, "dock_c": dock_c})
        indoor_k = indoor_c + KELVIN_OFFSET
        omega_indoor = self._infiltration_indoor_humidity_ratio(indoor_k, p_i)
        rho_i = air_density(indoor_k, p_i, omega_indoor)
        self._infiltration_state = TianInfiltrationState(density_kg_m3=rho_i)
        self._current_infiltration = zero_infiltration_result()
        self._current_infiltration["region_density_kg_m3"] = self._infiltration_state.density_kg_m3

    def advance_infiltration_disturbance(self, time_s: float, dt_s: float, previous_values: dict[str, float]) -> None:
        if not self._uses_tian_infiltration():
            return

        cfg = self._infiltration_cfg()
        indoor_c = self._infiltration_indoor_temperature_c(cfg, previous_values)
        outdoor_source = str(cfg.get("outdoor_source", "dock")).lower()
        if outdoor_source == "ambient":
            outdoor_c = float(self.cfg["boundary_conditions"]["ambient_c"])
        elif outdoor_source == "fixed":
            outdoor_c = float(cfg.get("outdoor_c", cfg.get("T_o_c", self.cfg["boundary_conditions"]["ambient_c"])))
        else:
            outdoor_c = float(previous_values.get("dock_c", self.cfg["boundary_conditions"]["dock_initial_c"]))

        p_i = float(cfg.get("P_i", cfg.get("indoor_pressure_pa", self.cfg["air_cycle"]["p_low_pa"])))
        p_o = float(cfg.get("P_o", cfg.get("outdoor_pressure_pa", p_i)))
        indoor_k = indoor_c + KELVIN_OFFSET
        omega_indoor = self._infiltration_indoor_humidity_ratio(indoor_k, p_i)

        self._current_infiltration = advance_tian_infiltration(
            cfg,
            self._infiltration_state,
            time_s=time_s,
            dt_s=dt_s,
            room_c=indoor_c,
            outdoor_c=outdoor_c,
            pressure_i_pa=p_i,
            pressure_o_pa=p_o,
            omega_room=omega_indoor,
        )

    def _constraint_penalty(self, unknowns: np.ndarray) -> np.ndarray:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock, dock_c = unknowns[:10]
        penalties = [
            max(0.0, -83.15 - room_c),
            max(0.0, room_c - 46.85),
            max(0.0, -50.0 - dock_c),
            max(0.0, dock_c - 46.85),
            max(0.0, -3.15 - sink_c),
            max(0.0, sink_c - 86.85),
            max(0.0, -73.15 - tevap_c),
            max(0.0, tevap_c - 46.85),
            max(0.0, 0.0 - tcond_c),
            max(0.0, tcond_c - 90.0),
            max(0.0, 1.0e-6 - m_ref_cascade),
            max(0.0, 1.0e-6 - m_ref_dock),
            max(0.0, self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock) - 5.0),
        ]
        residual_size = 10 + self._receiver_extra_residual_count()
        if self.high_pressure_receiver_enabled():
            receiver_mass = self._receiver_unknown_mass(unknowns)
            receiver_cfg = self._receiver_config()
            receiver_mass_max = float(receiver_cfg.get("mass_max_kg", receiver_cfg.get("maximum_mass_kg", 100.0)))
            penalties.extend(
                [
                    max(0.0, float(receiver_cfg.get("mass_min_kg", 0.0)) - receiver_mass),
                    max(0.0, receiver_mass - receiver_mass_max),
                ]
            )
        if self.lpr_subcooling_control_enabled():
            subcooling_k = self._lpr_subcooling_unknown(unknowns, 10)
            lpr_cfg = self._low_pressure_receiver_config()
            penalties.extend(
                [
                    max(0.0, float(lpr_cfg.get("subcooling_min_k", 0.0)) - subcooling_k),
                    max(0.0, subcooling_k - float(lpr_cfg.get("subcooling_max_k", 30.0))),
                ]
            )
        penalty_sum = sum(penalties)
        if penalty_sum < 1.0e-7:
            return np.zeros(residual_size, dtype=float)
        scale = 1.0e6
        p0 = 1.0e3 + scale * penalty_sum
        return np.full(residual_size, p0, dtype=float)

    def _base_room_load_w(self, time_s: float) -> float:
        bc = self.cfg["boundary_conditions"]
        if time_s < bc["load_step_time_s"]:
            return bc["load_before_w"]
        return bc["load_after_w"]

    def _base_dock_load_w(self, time_s: float) -> float:
        bc = self.cfg["boundary_conditions"]
        if time_s < bc["load_step_time_s"]:
            return bc["dock_load_before_w"]
        return bc["dock_load_after_w"]

    def _delayed_trapezoid_fraction(
        self,
        time_s: float,
        start_s: float,
        delay_s: float,
        ramp_s: float,
        hold_s: float,
    ) -> float:
        active_time_s = time_s - start_s - delay_s
        if active_time_s <= 0.0:
            return 0.0
        if ramp_s <= 0.0 and hold_s <= 0.0:
            return 1.0
        if ramp_s <= 0.0:
            return 1.0 if active_time_s <= hold_s else 0.0

        if active_time_s < ramp_s:
            return active_time_s / ramp_s

        if active_time_s < ramp_s + hold_s:
            return 1.0

        if active_time_s < 2.0 * ramp_s + hold_s:
            return max(0.0, (2.0 * ramp_s + hold_s - active_time_s) / ramp_s)

        return 0.0

    def infiltration_disturbance_w(self, time_s: float) -> dict[str, float]:
        cfg = self._infiltration_cfg()
        if not cfg.get("enabled", False):
            return zero_infiltration_result()

        if self._uses_tian_infiltration():
            return self._current_infiltration

        fraction = self._delayed_trapezoid_fraction(
            time_s,
            start_s=cfg["start_time_s"],
            delay_s=cfg.get("delay_s", 0.0),
            ramp_s=cfg.get("ramp_time_s", 30.0),
            hold_s=cfg.get("hold_time_s", 60.0),
        )
        magnitude_w = cfg.get("resolved_magnitude_w", cfg.get("magnitude_w", 0.0)) * fraction
        room_w, dock_w = apply_infiltration_load_application(cfg, magnitude_w)
        return {
            "room_w": room_w,
            "dock_w": dock_w,
            "q_m3_s": 0.0,
            "q_sensible_w": magnitude_w,
            "q_latent_w": 0.0,
            "q_total_w": magnitude_w,
            "cumulative_volume_m3": 0.0,
            "velocity_m_s": 0.0,
            "region_density_kg_m3": 0.0,
            "stage": 0.0,
            "door_open_fraction": fraction,
            "effective_length_m": 0.0,
            "effective_volume_m3": 0.0,
            "maximum_effective_length_m": 0.0,
        }

    def load_w(self, time_s: float) -> float:
        return (
            self._base_room_load_w(time_s)
            + self.infiltration_disturbance_w(time_s)["room_w"]
            + self._room_moisture.q_deposition_w
        )

    def _standalone_air_cycle_cold_loss_w(self, room_k: float) -> float:
        air_cfg = self.cfg.get("air_cycle", {})
        cold_load_cfg = air_cfg.get("cold_load", {})
        if not isinstance(cold_load_cfg, dict):
            cold_load_cfg = {}
        ua_w_k = float(cold_load_cfg.get("heat_leak_ua_w_k", air_cfg.get("cold_load_heat_leak_ua_w_k", 0.0)))
        if ua_w_k <= 0.0:
            return 0.0
        ambient_k = float(
            cold_load_cfg.get(
                "ambient_k",
                self.cfg.get("boundary_conditions", {}).get("ambient_c", 30.0) + KELVIN_OFFSET,
            )
        )
        return ua_w_k * max(ambient_k - float(room_k), 0.0)

    def dock_load_w(self, time_s: float) -> float:
        return self._base_dock_load_w(time_s) + self.infiltration_disturbance_w(time_s)["dock_w"]

    def _dock_evaporator_config(self) -> dict:
        vcc_cfg = self.cfg["vcc_cycle"]
        dock_cfg = dict(vcc_cfg.get("dock_evaporator", {}))
        dock_cfg.setdefault("ua_w_k", vcc_cfg.get("dock_evaporator_ua_w_k", 0.0))
        return dock_cfg

    def _expansion_valve_config(self, branch: str) -> dict:
        vcc_cfg = self.cfg["vcc_cycle"]
        base_cfg = dict(vcc_cfg.get("expansion_valve", {}))
        branch_cfg = dict(vcc_cfg.get("expansion_valves", {}).get(branch, {}))
        merged = {**base_cfg, **branch_cfg}
        merged.setdefault("opening", 0.5)
        if "flow_coefficient_m2" not in merged:
            merged.setdefault(
                "flow_coefficient_m2",
                base_cfg.get(
                    "flow_coefficient_m2",
                    base_cfg.get("flow_coefficient_kg_s_sqrt_pa_density", base_cfg.get("flow_coefficient_kg_s_pa", 0.0)),
                ),
            )
        return merged

    def startup_evaluation(self, unknowns: np.ndarray, time_s: float) -> tuple[np.ndarray, dict[str, float]]:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock, dock_c = unknowns[:10]
        receiver_mass_kg = self._receiver_unknown_mass(unknowns)
        subcooling_k = self._lpr_subcooling_unknown(unknowns, 10)
        m_ref = self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock)
        room_k = room_c + KELVIN_OFFSET
        sink_k = sink_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET
        tcond_k = tcond_c + KELVIN_OFFSET
        air_cfg = self.cfg["air_cycle"]
        vcc_cfg = self.cfg["vcc_cycle"]
        bc = self.cfg["boundary_conditions"]
        ambient_c = bc["ambient_c"]

        air = self._evaluate_air_cycle(room_k, t3_k, t4_k, t6_k)
        dock_evap = self._evaluate_dock_evaporator(dock_c, tevap_c)
        q_dock_evap = dock_evap["q_w"]
        infiltration = self.infiltration_disturbance_w(time_s)
        ref = self._evaluate_refrigerant_cycle(
            tevap_k,
            tcond_k,
            m_ref_cascade,
            m_ref_dock,
            air["q_cascade"],
            q_dock_evap,
            receiver_mass_kg,
            subcooling_k,
        )
        hx_uas = self._heat_exchanger_uas(air)

        t2_c = air["t2_k"] - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - tevap_c, t3_c - tevap_c)
        condenser_lmtd = positive_lmtd(tcond_c - ambient_c, tcond_c - sink_c)

        q_reg_ua = hx_uas["regenerator"]["ua_w_k"] * reg_lmtd
        q_cascade_ua = hx_uas["cascade"]["ua_w_k"] * cascade_lmtd
        q_cond_ua = hx_uas["condenser"]["ua_w_k"] * condenser_lmtd
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - ambient_c)
        if ref["uses_map_cooling_capacity_balance"]:
            cascade_balance = ref["q_evap_load_requested"] - ref["compressor_map_q_w"]
        else:
            cascade_balance = air["q_cascade"] - q_cascade_ua

        balances = [
            air["q_room"] - self.load_w(time_s),
            q_dock_evap - self.dock_load_w(time_s),
            ref["q_cond"] - sink_rejection,
            air["q_reg_hot"] - air["q_reg_cold"],
            air["q_reg_hot"] - q_reg_ua,
            cascade_balance,
            ref["q_cond"] - q_cond_ua,
            m_ref_cascade - ref["m_ref_valve_cascade"],
            m_ref_dock - ref["m_ref_valve_dock"],
            m_ref - ref["m_ref_compressor"],
        ]
        if self.high_pressure_receiver_enabled():
            balances.append(self._receiver_inlet_flow_kg_s(ref["m_ref_compressor"]) - ref["m_ref_valve"])

        metrics = {
            "room_c": room_c,
            "dock_c": dock_c,
            "sink_c": sink_c,
            "t2_c": t2_c,
            "t3_c": t3_c,
            "t4_c": t4_c,
            "t5_c": air["t5_k"] - KELVIN_OFFSET,
            "t6_c": t6_c,
            "t7_c": ref["t7_k"] - KELVIN_OFFSET,
            "tevap_c": tevap_c,
            "tcond_c": tcond_c,
            "m_ref_kg_s": m_ref,
            "m_ref_cascade_kg_s": m_ref_cascade,
            "m_ref_dock_kg_s": m_ref_dock,
            "m_air_kg_s": air["m_air"],
            "air_pressure_ratio": air["pressure_ratio"],
            "air_damper_opening": air["damper_opening"],
            "air_damper_resistance_head_coefficient": air["damper_resistance_head_coefficient"],
            "refrigerant_evaporating_pressure_pa": ref["p_evap"],
            "refrigerant_condensing_pressure_pa": ref["p_cond"],
            "refrigerant_pressure_ratio": ref["pressure_ratio"],
            "refrigerant_condensing_pressure_from_tcond_pa": ref["p_cond_from_tcond"],
            "refrigerant_pressure_ratio_from_tcond": ref["pressure_ratio_from_tcond"],
            "air_compressor_map_mass_flow_kg_s": air["m_air_map"],
            "infiltration_room_w": infiltration["room_w"],
            "infiltration_dock_w": infiltration["dock_w"],
            "humidity_ratio_room_kg_kg_da": air["humidity_ratio_room"],
            "humidity_ratio_supply_vapor_kg_kg_da": air["humidity_ratio_5_vapor"],
            "humidity_ratio_supply_ice_kg_kg_da": air["humidity_ratio_5_ice"],
            "ice_mass_flow_kg_s": air["ice_mass_flow"],
            "q_dock_w": q_dock_evap,
            "regenerator_ua_w_k": hx_uas["regenerator"]["ua_w_k"],
            "regenerator_ua_ref_w_k": hx_uas["regenerator"]["ua_ref_w_k"],
            "regenerator_air_flow_ratio": hx_uas["regenerator"]["flow_ratio"],
            "cascade_ua_w_k": hx_uas["cascade"]["ua_w_k"],
            "cascade_ua_ref_w_k": hx_uas["cascade"]["ua_ref_w_k"],
            "cascade_air_flow_ratio": hx_uas["cascade"]["flow_ratio"],
            "condenser_ua_w_k": hx_uas["condenser"]["ua_w_k"],
            "condenser_ua_ref_w_k": hx_uas["condenser"]["ua_ref_w_k"],
            "condenser_sink_flow_ratio": hx_uas["condenser"]["flow_ratio"],
            "dock_evaporator_ua_w_k": dock_evap["ua_w_k"],
            "dock_evaporator_ua_ref_w_k": dock_evap["ua_ref_w_k"],
            "dock_evaporator_lmtd_k": dock_evap["lmtd_k"],
            "dock_evaporator_air_outlet_c": dock_evap["air_outlet_c"],
            "dock_evaporator_air_delta_t_k": dock_evap["air_delta_t_k"],
            "dock_air_m_dot_kg_s": dock_evap["air_m_dot_kg_s"],
            "dock_air_m_dot_nominal_kg_s": dock_evap["air_m_dot_nominal_kg_s"],
            "dock_evaporator_ua_flow_exponent": dock_evap["ua_flow_exponent"],
            "q_cascade_refrigerant_w": ref["q_cascade_branch"],
            "q_dock_refrigerant_w": ref["q_dock_branch"],
            "q_evap_total_w": ref["q_evap_total"],
            "q_evap_load_requested_w": ref["q_evap_load_requested"],
            "q_map_capacity_balance_error_w": ref["q_map_capacity_balance_error"],
            "refrigerant_subcooling_k": ref["subcooling_k"],
            "refrigerant_superheat_k": ref["superheat_k"],
            "refrigerant_cascade_superheat_k": ref["superheat_cascade_k"],
            "refrigerant_dock_superheat_k": ref["superheat_dock_k"],
            "refrigerant_compressor_work_w": ref["w_ref_comp"],
            "refrigerant_compressor_isentropic_work_w": ref["w_ref_isentropic"],
            "refrigerant_q_condenser_stage_w": ref["q_condenser_stage"],
            "refrigerant_q_subcooler_w": ref["q_subcooler"],
            "refrigerant_q_oil_cooling_w": ref["q_oil_cooling_w"],
            "refrigerant_compressor_eta_is_target": ref["eta_is_target"],
            "refrigerant_compressor_eta_is_effective": ref["eta_is_effective"],
            "refrigerant_compressor_speed_rpm": ref["compressor_speed_rpm"],
            "refrigerant_compressor_map_capacity_w": ref["compressor_map_q_w"],
            "refrigerant_compressor_volumetric_efficiency": ref["compressor_eta_v"],
            "refrigerant_compressor_suction_density_kg_m3": ref["compressor_suction_density_kg_m3"],
            "refrigerant_compressor_displacement_m3_per_rev": ref["compressor_displacement_m3_per_rev"],
            "low_pressure_receiver_enabled": ref["low_pressure_receiver_enabled"],
            "refrigerant_evaporators_series": ref["evaporator_arrangement_series"],
            "refrigerant_evaporator_outlet_c": ref["t7_evaporator_out_k"] - KELVIN_OFFSET,
            "air_compressor_actual_head_j_kg": air["compressor_head_actual"],
            "air_compressor_actual_head_m": air["compressor_head_actual"] / 9.80665,
            "air_compressor_isentropic_head_j_kg": air["compressor_head_is"],
            "air_compressor_isentropic_head_m": air["compressor_head_is"] / 9.80665,
            "air_compressor_eta_is_target": air["compressor_eta_is_target"],
            "air_compressor_eta_is_effective": air["compressor_eta_is_effective"],
            "receiver_enabled": ref["receiver_enabled"],
            "receiver_mass_kg": ref["receiver_mass_kg"],
            "receiver_volume_m3": ref["receiver_volume_m3"],
            "receiver_liquid_mass_kg": ref["receiver_liquid_mass_kg"],
            "receiver_vapor_mass_kg": ref["receiver_vapor_mass_kg"],
            "receiver_liquid_volume_m3": ref["receiver_liquid_volume_m3"],
            "receiver_liquid_fill_fraction": ref["receiver_liquid_fill_fraction"],
            "receiver_liquid_fill_fraction_raw": ref["receiver_liquid_fill_fraction_raw"],
            "receiver_overfill_mass_kg": ref["receiver_overfill_mass_kg"],
            "receiver_liquid_feed_fraction": ref["receiver_liquid_feed_fraction"],
            "receiver_starvation_flow_multiplier": ref["receiver_starvation_flow_multiplier"],
            "receiver_outlet_enthalpy_j_kg": ref["receiver_outlet_enthalpy_j_kg"],
            "receiver_outlet_density_kg_m3": ref["receiver_outlet_density_kg_m3"],
            "receiver_inlet_m_dot_kg_s": ref["receiver_inlet_m_dot_kg_s"],
            "receiver_outlet_m_dot_kg_s": ref["m_ref_valve"],
        }
        return np.asarray(balances, dtype=float), metrics

    def _pressure_from_isentropic_head(
        self,
        p1: float,
        x_room: float,
        entropy_j_kg_da_k: float,
        h1_j_kg_da: float,
        target_head_j_kg: float,
    ) -> float:
        air_cfg = self.cfg["air_cycle"]
        flow_cfg = air_cfg["compressor_mass_flow"]
        lower_pr = float(flow_cfg.get("pressure_ratio_min", air_cfg.get("pressure_ratio_min", 1.001)))
        upper_pr = float(flow_cfg.get("pressure_ratio_max", air_cfg.get("pressure_ratio_max", 2.0)))

        def residual(pressure_ratio: float) -> float:
            state2s = state_at_entropy(p1 * pressure_ratio, x_room, entropy_j_kg_da_k)
            return state2s.enthalpy_j_kg_da - h1_j_kg_da - target_head_j_kg

        lo = max(lower_pr, 1.000001)
        hi = max(upper_pr, lo)
        f_lo = residual(lo)
        f_hi = residual(hi)
        if f_lo * f_hi > 0.0:
            return p1 * (lo if abs(f_lo) <= abs(f_hi) else hi)

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if abs(f_mid) <= 1.0e-7 or abs(hi - lo) <= 1.0e-9:
                return p1 * mid
            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return p1 * (0.5 * (lo + hi))

    def _evaluate_air_cycle(self, room_k: float, t3_k: float, t4_k: float, t6_k: float) -> dict[str, float]:
        air_cfg = self.cfg["air_cycle"]
        p1 = air_cfg["p_low_pa"]
        x_room = self._room_humidity_ratio(room_k, p1)

        state_room = humid_air_state(room_k, p1, x_room)
        state1 = humid_air_state(t6_k, p1, x_room)
        h1 = state1.enthalpy_j_kg_da
        rho1 = state1.dry_air_density_kg_m3
        flow_cfg = air_cfg["compressor_mass_flow"]
        flow_model = flow_cfg.get("model", "polynomial_volumetric_flow_head")
        compressor_eta_is = float(air_cfg["compressor_eta_is"])
        map_m_air = 0.0
        compressor_head_actual: float | None = None
        if flow_model == "polynomial_volumetric_flow_head_speed_damper":
            head_m, volumetric_flow_m3_s = self._air_damper_operating_point(flow_cfg)
            m_air = volumetric_flow_m3_s * rho1
            map_m_air = m_air
            compressor_head_actual = head_m * 9.80665
            p2 = self._pressure_from_isentropic_head(
                p1,
                x_room,
                state1.entropy_j_kg_da_k,
                h1,
                compressor_head_actual * compressor_eta_is,
            )
            state2s = state_at_entropy(p2, x_room, state1.entropy_j_kg_da_k)
            h2s = state2s.enthalpy_j_kg_da
            compressor_head_is = h2s - h1
        elif flow_model == "polynomial_volumetric_flow_head_speed_constant_mass_flow":
            m_air = float(flow_cfg["fixed_m_dot_kg_s"])
            if "fixed_discharge_pressure_pa" in flow_cfg:
                p2 = float(flow_cfg["fixed_discharge_pressure_pa"])
                map_m_air = m_air
            elif flow_cfg.get("use_configured_pressure_ratio", False):
                p2 = p1 * air_cfg["pressure_ratio"]
                map_m_air = m_air
            else:
                head_m, map_m_air = head_from_mass_flow(flow_cfg, m_air, rho1)
                compressor_head_actual = head_m * 9.80665
                p2 = self._pressure_from_isentropic_head(
                    p1,
                    x_room,
                    state1.entropy_j_kg_da_k,
                    h1,
                    compressor_head_actual * compressor_eta_is,
                )
            state2s = state_at_entropy(p2, x_room, state1.entropy_j_kg_da_k)
            h2s = state2s.enthalpy_j_kg_da
            compressor_head_is = h2s - h1
            if compressor_head_actual is None:
                compressor_head_actual = compressor_head_is / max(compressor_eta_is, 1.0e-6)
        elif flow_model in {"lumped_screw_compressor", "lumped_screw_constant_mass_flow"}:
            m_air = float(flow_cfg.get("fixed_m_dot_kg_s", flow_cfg.get("mass_flow_kg_s", 0.0)))
            if m_air <= 0.0:
                raise ValueError("lumped_screw_compressor requires fixed_m_dot_kg_s > 0.")
            map_m_air = m_air

            if "discharge_pressure_pa" in flow_cfg:
                p2 = float(flow_cfg["discharge_pressure_pa"])
            elif "high_side_pressure_pa" in flow_cfg:
                p2 = float(flow_cfg["high_side_pressure_pa"])
            elif "expander_inlet_pressure_pa" in flow_cfg:
                p2 = float(flow_cfg["expander_inlet_pressure_pa"])
            elif "discharge_pressure_min_pa" in flow_cfg and "discharge_pressure_max_pa" in flow_cfg:
                p2 = 0.5 * (float(flow_cfg["discharge_pressure_min_pa"]) + float(flow_cfg["discharge_pressure_max_pa"]))
            elif "expander_inlet_pressure_min_pa" in flow_cfg and "expander_inlet_pressure_max_pa" in flow_cfg:
                p2 = 0.5 * (
                    float(flow_cfg["expander_inlet_pressure_min_pa"])
                    + float(flow_cfg["expander_inlet_pressure_max_pa"])
                )
            else:
                p2 = p1 * float(flow_cfg.get("pressure_ratio", air_cfg["pressure_ratio"]))

            state2s = state_at_entropy(p2, x_room, state1.entropy_j_kg_da_k)
            h2s = state2s.enthalpy_j_kg_da
            compressor_head_is = h2s - h1
            compressor_head_actual = compressor_head_is / max(compressor_eta_is, 1.0e-6)
        else:
            p2 = p1 * air_cfg["pressure_ratio"]
            state2s = state_at_entropy(p2, x_room, state1.entropy_j_kg_da_k)
            h2s = state2s.enthalpy_j_kg_da
            compressor_head_is = h2s - h1
            compressor_head_actual = compressor_head_is / max(compressor_eta_is, 1.0e-6)
            m_air = mass_flow_from_actual_head(flow_cfg, compressor_head_actual, rho1)
            map_m_air = m_air
        if compressor_head_actual is None:
            compressor_head_actual = compressor_head_is / max(compressor_eta_is, 1.0e-6)
        h2 = h1 + compressor_head_actual
        state2 = state_at_enthalpy(p2, x_room, h2)
        t2_k = state2.temperature_k
        w_air_comp = m_air * (h2 - h1)

        state3 = humid_air_state(t3_k, p2, x_room)
        state4 = humid_air_state(t4_k, p2, x_room)
        state5s = state_at_entropy(p1, x_room, state4.entropy_j_kg_da_k)
        h4 = state4.enthalpy_j_kg_da
        h5s = state5s.enthalpy_j_kg_da
        h5 = turbine_actual_enthalpy(h4, h5s, air_cfg["turbine_eta_is"])
        state5 = state_at_enthalpy(p1, x_room, h5)
        t5_k = state5.temperature_k

        h3 = state3.enthalpy_j_kg_da
        h6 = state1.enthalpy_j_kg_da
        h_room = state_room.enthalpy_j_kg_da

        return {
            "p1": p1,
            "p2": p2,
            "pressure_ratio": p2 / p1,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "h5": h5,
            "h6": h6,
            "h_room": h_room,
            "t2_k": t2_k,
            "t5_k": t5_k,
            "m_air": m_air,
            "m_air_map": map_m_air,
            "damper_opening": float(flow_cfg.get("damper_opening", 1.0)),
            "damper_resistance_head_coefficient": float(flow_cfg.get("damper_resistance_head_coefficient", 0.0)),
            "rho1": rho1,
            "rho1_mixture": state1.mixture_density_kg_m3,
            "humidity_ratio_room": x_room,
            "humidity_ratio_5_vapor": state5.vapor_humidity_ratio,
            "humidity_ratio_5_ice": state5.ice_humidity_ratio,
            "ice_mass_flow": m_air * state5.ice_humidity_ratio,
            "compressor_head_is": compressor_head_is,
            "compressor_head_actual": compressor_head_actual,
            "compressor_eta_is_target": compressor_eta_is,
            "compressor_eta_is_effective": compressor_head_is / max(compressor_head_actual, 1.0e-9),
            "q_cascade": m_air * (h2 - h3),
            "q_reg_hot": m_air * (h3 - h4),
            "q_reg_cold": m_air * (h6 - h_room),
            "q_room": m_air * (h_room - h5),
            "w_air_comp": w_air_comp,
            "w_air_turb": m_air * (h4 - h5),
        }

    @staticmethod
    def _air_cycle_input_power(air: dict[str, float], air_cfg: dict) -> float:
        recover_turb = air_cfg.get("recover_turbine_work", True)
        w_turb_recovered = air["w_air_turb"] if recover_turb else 0.0
        return (air["w_air_comp"] - w_turb_recovered) / max(
            air_cfg.get("combined_drive_efficiency", 1.0),
            1.0e-6,
        )

    @staticmethod
    def _fixed_air_cycle_after_cooler_outlet_c(air_cfg: dict) -> float | None:
        after_cooler_cfg = air_cfg.get("after_cooler", {})
        if not isinstance(after_cooler_cfg, dict):
            after_cooler_cfg = {}

        for cfg in (after_cooler_cfg, air_cfg):
            if "fixed_outlet_c" in cfg:
                return float(cfg["fixed_outlet_c"])
            if "fixed_outlet_k" in cfg:
                return float(cfg["fixed_outlet_k"]) - KELVIN_OFFSET
            if "fixed_after_cooler_outlet_c" in cfg:
                return float(cfg["fixed_after_cooler_outlet_c"])
            if "fixed_after_cooler_outlet_k" in cfg:
                return float(cfg["fixed_after_cooler_outlet_k"]) - KELVIN_OFFSET
        return None

    def _air_damper_operating_point(self, flow_cfg: dict) -> tuple[float, float]:
        head_min_m = float(flow_cfg["head_min_m"])
        head_max_m = float(flow_cfg["head_max_m"])
        opening = float(flow_cfg.get("damper_opening", 0.5))
        opening_min = float(flow_cfg.get("damper_opening_min", 0.05))
        opening_max = float(flow_cfg.get("damper_opening_max", 1.0))
        opening = min(max(opening, opening_min), opening_max)
        resistance = float(flow_cfg.get("damper_resistance_head_coefficient", 0.0))
        static_head_m = float(flow_cfg.get("system_static_head_m", 0.0))

        if resistance <= 0.0:
            head = min(max(float(flow_cfg.get("operating_head_m", head_min_m)), head_min_m), head_max_m)
            return head, volumetric_flow_from_head(flow_cfg, head)

        def residual(head_m: float) -> float:
            q_m3_s = volumetric_flow_from_head(flow_cfg, head_m)
            required_head_m = static_head_m + resistance * (q_m3_s / max(opening, 1.0e-9)) ** 2
            return head_m - required_head_m

        sample_heads = np.linspace(head_min_m, head_max_m, 200)
        sample_residuals = np.array([residual(float(head_m)) for head_m in sample_heads], dtype=float)
        best_idx = int(np.argmin(np.abs(sample_residuals)))

        bracket: tuple[float, float] | None = None
        for idx in range(len(sample_heads) - 1):
            f_lo = sample_residuals[idx]
            f_hi = sample_residuals[idx + 1]
            if f_lo == 0.0:
                head = float(sample_heads[idx])
                return head, volumetric_flow_from_head(flow_cfg, head)
            if f_lo * f_hi <= 0.0:
                bracket = (float(sample_heads[idx]), float(sample_heads[idx + 1]))
                break

        if bracket is None:
            head = float(sample_heads[best_idx])
            return head, volumetric_flow_from_head(flow_cfg, head)

        lo, hi = bracket
        f_lo = residual(lo)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if abs(f_mid) <= 1.0e-9 or abs(hi - lo) <= 1.0e-7:
                return mid, volumetric_flow_from_head(flow_cfg, mid)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo = mid
                f_lo = f_mid

        head = 0.5 * (lo + hi)
        return head, volumetric_flow_from_head(flow_cfg, head)

    def _branch_valve_flow(self, valve_cfg: dict, p_cond: float, p_evap: float, inlet_density_kg_m3: float) -> float:
        coefficient = expansion_valve_flow_coefficient(valve_cfg)
        opening = float(valve_cfg.get("opening", 0.0))
        return coefficient * opening * expansion_valve_flow_factor(p_cond, p_evap, inlet_density_kg_m3)

    def _evaluate_refrigerant_cycle(
        self,
        tevap_k: float,
        tcond_k: float,
        m_ref_cascade: float,
        m_ref_dock: float,
        q_cascade: float,
        q_dock: float,
        receiver_mass_kg: float | None = None,
        subcooling_override_k: float | None = None,
    ) -> dict[str, float]:
        vcc_cfg = self.cfg["vcc_cycle"]
        p_evap = p_sat(tevap_k, self.ref_fluid)
        p_cond_from_tcond = p_sat(tcond_k, self.ref_fluid)
        p_cond = p_cond_from_tcond
        configured_subcooling_k = float(vcc_cfg["subcooling_k"])
        if self._hpr_forces_saturated_liquid():
            subcooling_k = 0.0
        elif subcooling_override_k is not None:
            subcooling_k = max(float(subcooling_override_k), 0.0)
        else:
            subcooling_k = configured_subcooling_k
        t9_k = tcond_k - subcooling_k
        h_sat_liq_cond = props_si("H", "T", tcond_k, "Q", 0.0, self.ref_fluid)
        h9 = h_sat_liq_cond if self._hpr_forces_saturated_liquid() else h_refrigerant_liquid(
            t9_k,
            p_cond_from_tcond,
            self.ref_fluid,
            subcooling_k,
        )
        receiver = self._evaluate_high_pressure_receiver(receiver_mass_kg, p_cond, tcond_k, h9)
        h10 = receiver["outlet_enthalpy_j_kg"]
        compressor_cfg = vcc_cfg["compressor"]
        compressor_speed_rpm = float(compressor_cfg.get("speed_rpm", 0.0))
        compressor_eta_is = ammonia_compressor_eta_is(
            compressor_cfg,
            tcond_k,
            tevap_k,
            compressor_speed_rpm,
            self.ref_fluid,
        )
        q_evap_load_requested = q_cascade + q_dock
        m_ref_cascade = max(float(m_ref_cascade), 1.0e-12)
        m_ref_dock = max(float(m_ref_dock), 1.0e-12)
        series_evaporators = self.vcc_evaporators_are_series()
        m_ref = self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock)
        compressor_map_q_w = 0.0
        m_ref_compressor = m_ref
        w_ref_comp = 0.0
        q_oil_cooling_w = 0.0
        compressor_eta_v = 1.0
        compressor_suction_density = 0.0
        compressor_displacement = float(compressor_cfg.get("displacement_m3_per_rev", 0.0))
        uses_map_capacity_balance = False
        compressor_model = compressor_cfg.get("model", "simple_isentropic")
        if compressor_model == "bitzer_variable_speed_map":
            compressor_map = ammonia_compressor_map(
                tcond_k,
                tevap_k,
                compressor_speed_rpm,
                check_range=bool(compressor_cfg.get("check_range", False)),
            )
            compressor_map_q_w = compressor_map["Q_W"]
            w_ref_comp = compressor_map["P_W"]
            m_ref_compressor = compressor_map["mdot_kg_s"]
            uses_map_capacity_balance = compressor_uses_map_cooling_capacity_balance(compressor_cfg)
        elif compressor_model in {
            "screw_pressure_ratio_polynomial",
            "screw_polynomial",
            "pressure_ratio_polynomial_screw",
        }:
            uses_map_capacity_balance = False
        elif compressor_model in {"positive_displacement_clearance", "positive_displacement"}:
            uses_map_capacity_balance = False
        evap_total = compressor_map_q_w if uses_map_capacity_balance else q_evap_load_requested

        branch_scale = evap_total / max(q_evap_load_requested, 1.0e-9)
        q_cascade_branch = q_cascade * branch_scale
        q_dock_branch = q_dock * branch_scale
        cascade_valve_cfg = self._expansion_valve_config("cascade")
        dock_valve_cfg = self._expansion_valve_config("dock")
        has_branch_valves = bool(vcc_cfg.get("expansion_valves"))

        p_cond_for_valves = p_cond
        if series_evaporators:
            h_dock_out = h10 + q_dock_branch / max(m_ref, 1.0e-9)
            h_cascade_out = h_dock_out + q_cascade_branch / max(m_ref, 1.0e-9)
            h7_evaporator_out = h_cascade_out
        else:
            h_cascade_out = h10 + q_cascade_branch / max(m_ref_cascade, 1.0e-9)
            h_dock_out = h10 + q_dock_branch / max(m_ref_dock, 1.0e-9)
            h7_evaporator_out = (m_ref_cascade * h_cascade_out + m_ref_dock * h_dock_out) / max(m_ref, 1.0e-9)
        t7_evaporator_out_k = props_si("T", "P", p_evap, "H", h7_evaporator_out, self.ref_fluid)
        if self._compressor_uses_saturated_suction():
            h7 = props_si("H", "P", p_evap, "Q", 1.0, self.ref_fluid)
            t7_k = props_si("T", "P", p_evap, "Q", 1.0, self.ref_fluid)
            s7 = props_si("S", "P", p_evap, "Q", 1.0, self.ref_fluid)
        else:
            h7 = h7_evaporator_out
            t7_k = t7_evaporator_out_k
            s7 = props_si("S", "P", p_evap, "H", h7, self.ref_fluid)
        t_cascade_out_k = props_si("T", "P", p_evap, "H", h_cascade_out, self.ref_fluid)
        t_dock_out_k = props_si("T", "P", p_evap, "H", h_dock_out, self.ref_fluid)

        if has_branch_valves:
            for _ in range(3):
                if compressor_model == "bitzer_variable_speed_map" and compressor_uses_map_power_pressure_lift(compressor_cfg):
                    p_cond_for_valves = compressor_discharge_pressure_from_power(
                        p_evap,
                        h7,
                        s7,
                        m_ref_compressor,
                        w_ref_comp,
                        compressor_eta_is,
                        self.ref_fluid,
                        pressure_ratio_min=float(compressor_cfg.get("pressure_ratio_min", 1.000001)),
                        pressure_ratio_max=compressor_cfg.get("pressure_ratio_max"),
                    )
            p_cond = p_cond_for_valves

        if self._compressor_uses_saturated_suction():
            superheat_k = 0.0
            superheat_cascade_k = 0.0
            superheat_dock_k = 0.0
        else:
            superheat_k = t7_k - tevap_k
            superheat_cascade_k = t_cascade_out_k - tevap_k
            superheat_dock_k = t_dock_out_k - tevap_k
        if compressor_model == "bitzer_variable_speed_map":
            if compressor_uses_map_power_pressure_lift(compressor_cfg) and not has_branch_valves:
                p_cond = compressor_discharge_pressure_from_power(
                    p_evap,
                    h7,
                    s7,
                    m_ref_compressor,
                    w_ref_comp,
                    compressor_eta_is,
                    self.ref_fluid,
                    pressure_ratio_min=float(compressor_cfg.get("pressure_ratio_min", 1.000001)),
                    pressure_ratio_max=compressor_cfg.get("pressure_ratio_max"),
                )
        elif compressor_model in {
            "screw_pressure_ratio_polynomial",
            "screw_polynomial",
            "pressure_ratio_polynomial_screw",
        }:
            pressure_ratio = p_cond / max(p_evap, 1.0e-9)
            screw_map = screw_compressor_pressure_ratio_map(compressor_cfg, pressure_ratio)
            compressor_eta_is = screw_map["eta_is"]
            compressor_eta_v = screw_map["eta_v"]
            q_oil_cooling_w = screw_map["q_oil_w"]
            if compressor_displacement <= 0.0:
                swept_volume_m3_h = compressor_cfg.get("displacement_m3_h", compressor_cfg.get("swept_volume_m3_h"))
                nominal_speed_rpm = compressor_cfg.get("nominal_speed_rpm", compressor_cfg.get("design_speed_rpm"))
                if swept_volume_m3_h is not None and nominal_speed_rpm is not None:
                    compressor_displacement = float(swept_volume_m3_h) / max(float(nominal_speed_rpm) * 60.0, 1.0e-12)
            compressor_suction_density = props_si("D", "P", p_evap, "H", h7, self.ref_fluid)
            m_ref_compressor = compressor_mass_flow_positive_displacement(
                compressor_suction_density,
                compressor_speed_rpm,
                compressor_displacement,
                compressor_eta_v,
            )
            h8s_for_work = props_si("H", "P", p_cond, "S", s7, self.ref_fluid)
            w_ref_comp = m_ref_compressor * (h8s_for_work - h7) / max(compressor_eta_is, 1.0e-6)
        elif compressor_model in {"positive_displacement_clearance", "positive_displacement"}:
            pressure_ratio = p_cond / max(p_evap, 1.0e-9)
            compressor_suction_density = props_si("D", "P", p_evap, "H", h7, self.ref_fluid)
            compressor_eta_v = compressor_volumetric_efficiency_clearance(
                pressure_ratio,
                float(compressor_cfg.get("clearance_factor", 0.05)),
                float(compressor_cfg.get("polytropic_exponent", 1.25)),
                float(compressor_cfg.get("eta_v_min", 0.3)),
                float(compressor_cfg.get("eta_v_max", 1.0)),
            )
            m_ref_compressor = compressor_mass_flow_positive_displacement(
                compressor_suction_density,
                compressor_speed_rpm,
                compressor_displacement,
                compressor_eta_v,
            )
            h8s_for_work = props_si("H", "P", p_cond, "S", s7, self.ref_fluid)
            w_ref_comp = m_ref_compressor * (h8s_for_work - h7) / max(compressor_eta_is, 1.0e-6)
        else:
            h8s_for_work = props_si("H", "P", p_cond, "S", s7, self.ref_fluid)
            w_ref_comp = float(compressor_cfg.get("work_w", m_ref * (h8s_for_work - h7) / max(compressor_eta_is, 1.0e-6)))
        h8s = props_si("H", "P", p_cond, "S", s7, self.ref_fluid)
        w_ref_isentropic = m_ref_compressor * (h8s - h7)
        h8 = h7 + w_ref_comp / max(m_ref_compressor, 1.0e-6)
        t8_k = props_si("T", "P", p_cond, "H", h8, self.ref_fluid)
        q_cond = evap_total + w_ref_comp
        q_subcooler = max(m_ref * max(h_sat_liq_cond - h10, 0.0), 0.0)
        q_condenser_stage = max(q_cond - q_subcooler, 0.0)
        valve_inlet_density = receiver["outlet_density_kg_m3"]
        receiver_flow_multiplier = float(receiver["starvation_flow_multiplier"])
        if series_evaporators:
            m_ref_valve = (
                self._branch_valve_flow(cascade_valve_cfg, p_cond, p_evap, valve_inlet_density)
                * receiver_flow_multiplier
            )
            m_ref_valve_cascade = m_ref_valve
            m_ref_valve_dock = m_ref_valve
        elif has_branch_valves:
            m_ref_valve_cascade = (
                self._branch_valve_flow(cascade_valve_cfg, p_cond, p_evap, valve_inlet_density)
                * receiver_flow_multiplier
            )
            m_ref_valve_dock = (
                self._branch_valve_flow(dock_valve_cfg, p_cond, p_evap, valve_inlet_density)
                * receiver_flow_multiplier
            )
            m_ref_valve = m_ref_valve_cascade + m_ref_valve_dock
        else:
            valve_cfg = vcc_cfg["expansion_valve"]
            m_ref_valve = self._branch_valve_flow(valve_cfg, p_cond, p_evap, valve_inlet_density) * receiver_flow_multiplier
            m_ref_valve_cascade = m_ref_valve
            m_ref_valve_dock = 0.0
        return {
            "p_evap": p_evap,
            "p_cond": p_cond,
            "h9": h9,
            "h10": h10,
            "low_pressure_receiver_enabled": float(self.low_pressure_receiver_enabled()),
            "receiver_enabled": receiver["enabled"],
            "receiver_mass_kg": receiver["mass_kg"],
            "receiver_volume_m3": receiver["volume_m3"],
            "receiver_liquid_mass_kg": receiver["liquid_mass_kg"],
            "receiver_vapor_mass_kg": receiver["vapor_mass_kg"],
            "receiver_liquid_volume_m3": receiver["liquid_volume_m3"],
            "receiver_liquid_fill_fraction": receiver["liquid_fill_fraction"],
            "receiver_liquid_fill_fraction_raw": receiver["liquid_fill_fraction_raw"],
            "receiver_overfill_mass_kg": receiver["overfill_mass_kg"],
            "receiver_liquid_feed_fraction": receiver["liquid_feed_fraction"],
            "receiver_starvation_flow_multiplier": receiver["starvation_flow_multiplier"],
            "receiver_outlet_enthalpy_j_kg": receiver["outlet_enthalpy_j_kg"],
            "receiver_outlet_density_kg_m3": receiver["outlet_density_kg_m3"],
            "receiver_inlet_m_dot_kg_s": self._receiver_inlet_flow_kg_s(m_ref_compressor),
            "h7": h7,
            "h7_evaporator_out": h7_evaporator_out,
            "t7_k": t7_k,
            "t7_evaporator_out_k": t7_evaporator_out_k,
            "t8_k": t8_k,
            "pressure_ratio": p_cond / max(p_evap, 1.0e-9),
            "p_cond_from_tcond": p_cond_from_tcond,
            "pressure_ratio_from_tcond": p_cond_from_tcond / max(p_evap, 1.0e-9),
            "subcooling_k": subcooling_k,
            "superheat_k": superheat_k,
            "superheat_cascade_k": superheat_cascade_k,
            "superheat_dock_k": superheat_dock_k,
            "h_cascade_out": h_cascade_out,
            "h_dock_out": h_dock_out,
            "t_cascade_out_k": t_cascade_out_k,
            "t_dock_out_k": t_dock_out_k,
            "q_cascade_branch": q_cascade_branch,
            "q_dock_branch": q_dock_branch,
            "q_evap_total": evap_total,
            "q_evap_load_requested": q_evap_load_requested,
            "q_map_capacity_balance_error": q_evap_load_requested - compressor_map_q_w,
            "uses_map_cooling_capacity_balance": uses_map_capacity_balance,
            "q_cond": q_cond,
            "q_condenser_stage": q_condenser_stage,
            "q_subcooler": q_subcooler,
            "w_ref_comp": w_ref_comp,
            "q_oil_cooling_w": q_oil_cooling_w,
            "w_ref_isentropic": w_ref_isentropic,
            "eta_is_effective": w_ref_isentropic / max(w_ref_comp, 1.0e-6),
            "eta_is_target": compressor_eta_is,
            "m_ref_valve": m_ref_valve,
            "m_ref_valve_cascade": m_ref_valve_cascade,
            "m_ref_valve_dock": m_ref_valve_dock,
            "m_ref_cascade": m_ref_cascade,
            "m_ref_dock": m_ref_dock,
            "m_ref_total": m_ref,
            "m_ref_compressor": m_ref_compressor,
            "evaporator_arrangement_series": float(series_evaporators),
            "compressor_speed_rpm": compressor_speed_rpm,
            "compressor_map_q_w": compressor_map_q_w,
            "compressor_eta_v": compressor_eta_v,
            "compressor_suction_density_kg_m3": compressor_suction_density,
            "compressor_displacement_m3_per_rev": compressor_displacement,
            "cascade_valve_opening": float(cascade_valve_cfg.get("opening", 0.0)),
            "dock_valve_opening": float(dock_valve_cfg.get("opening", 0.0)),
        }

    def _branch_holdup_outputs(self, ref: dict[str, float], tevap_k: float, time_s: float) -> dict[str, float]:
        if self._compressor_uses_saturated_suction():
            return {
                "superheat_k": 0.0,
                "superheat_cascade_k": 0.0,
                "superheat_dock_k": 0.0,
            }
        holdup_cfg = self.cfg["vcc_cycle"].get("branch_holdup", {})
        tau_s = float(holdup_cfg.get("outlet_enthalpy_time_constant_s", 0.0))
        raw_h = {
            "cascade": float(ref["h_cascade_out"]),
            "dock": float(ref["h_dock_out"]),
        }
        if tau_s <= 0.0 or self._branch_holdup_time_s is None:
            self._branch_holdup_h = dict(raw_h)
        else:
            dt_s = max(float(time_s) - self._branch_holdup_time_s, 0.0)
            alpha = min(max(dt_s / tau_s, 0.0), 1.0)
            for branch, h_target in raw_h.items():
                h_previous = float(self._branch_holdup_h.get(branch, h_target))
                self._branch_holdup_h[branch] = h_previous + alpha * (h_target - h_previous)
        self._branch_holdup_time_s = float(time_s)

        h_cascade = float(self._branch_holdup_h["cascade"])
        h_dock = float(self._branch_holdup_h["dock"])
        m_cascade = max(float(ref["m_ref_cascade"]), 1.0e-12)
        m_dock = max(float(ref["m_ref_dock"]), 1.0e-12)
        if ref.get("evaporator_arrangement_series", 0.0):
            h_mixed = h_cascade
        else:
            h_mixed = (m_cascade * h_cascade + m_dock * h_dock) / max(m_cascade + m_dock, 1.0e-12)

        t_cascade_k = props_si("T", "P", ref["p_evap"], "H", h_cascade, self.ref_fluid)
        t_dock_k = props_si("T", "P", ref["p_evap"], "H", h_dock, self.ref_fluid)
        t_mixed_k = props_si("T", "P", ref["p_evap"], "H", h_mixed, self.ref_fluid)
        return {
            "superheat_k": t_mixed_k - tevap_k,
            "superheat_cascade_k": t_cascade_k - tevap_k,
            "superheat_dock_k": t_dock_k - tevap_k,
        }

    def _evaluate_dock_evaporator(self, dock_c: float, tevap_c: float) -> dict[str, float]:
        dock_cfg = self._dock_evaporator_config()
        ua_ref_w_k = float(dock_cfg.get("ua_w_k", dock_cfg.get("ua_ref_w_k", 0.0)))
        ua_w_k = ua_ref_w_k
        ua_exponent = DYNAMIC_UA_DEFAULT_EXPONENTS["dock_evaporator"]
        model = str(dock_cfg.get("model", "fixed_ua")).lower()
        air_m_dot = 0.0
        air_flow_ratio = 0.0
        air_m_dot_ref = 0.0
        lmtd_k = max(dock_c - tevap_c, 0.0)
        air_outlet_c = dock_c
        air_delta_t_k = 0.0
        air_cp = float(dock_cfg.get("air_cp_j_kg_k", CP_DOCK_AIR))

        if model in {"ua_lmtd_air", "air_lmtd", "lmtd_air"}:
            air_m_dot = float(dock_cfg.get("air_m_dot_kg_s", dock_cfg.get("design_air_m_dot_kg_s", 0.0)))
            air_m_dot_min = float(dock_cfg.get("air_m_dot_min_kg_s", 0.0))
            air_m_dot_max = float(dock_cfg.get("air_m_dot_max_kg_s", max(air_m_dot, air_m_dot_min)))
            air_m_dot_max = max(air_m_dot_max, air_m_dot_min)
            air_m_dot = float(np.clip(air_m_dot, air_m_dot_min, air_m_dot_max))
            air_m_dot_ref = max(
                float(dock_cfg.get("nominal_air_m_dot_kg_s", dock_cfg.get("design_air_m_dot_kg_s", air_m_dot))),
                1.0e-9,
            )
            ua_result = self._scaled_ua_result("dock_evaporator", ua_ref_w_k, air_m_dot, air_m_dot_ref, dock_cfg)
            ua_w_k = ua_result["ua_w_k"]
            ua_exponent = ua_result["exponent"]
            air_flow_ratio = ua_result["flow_ratio"]

            delta_t_in = max(dock_c - tevap_c, 0.0)
            mcp = air_m_dot * air_cp
            if delta_t_in > 0.0 and ua_w_k > 0.0 and mcp > 0.0:
                ntu = ua_w_k / mcp
                air_outlet_c = tevap_c + delta_t_in * float(np.exp(-ntu))
                air_delta_t_k = max(dock_c - air_outlet_c, 0.0)
                q_w = mcp * air_delta_t_k
                lmtd_k = positive_lmtd(delta_t_in, max(air_outlet_c - tevap_c, 0.0))
            else:
                q_w = 0.0
        else:
            configured_air_m_dot = self._first_positive_config_value(dock_cfg, ("air_m_dot_kg_s", "design_air_m_dot_kg_s"))
            if configured_air_m_dot is not None:
                air_m_dot = configured_air_m_dot
                air_m_dot_ref = max(
                    float(dock_cfg.get("nominal_air_m_dot_kg_s", dock_cfg.get("design_air_m_dot_kg_s", air_m_dot))),
                    1.0e-9,
                )
                ua_result = self._scaled_ua_result("dock_evaporator", ua_ref_w_k, air_m_dot, air_m_dot_ref, dock_cfg)
                ua_w_k = ua_result["ua_w_k"]
                ua_exponent = ua_result["exponent"]
                air_flow_ratio = ua_result["flow_ratio"]
            q_w = ua_w_k * max(dock_c - tevap_c, 0.0)

        return {
            "q_w": q_w,
            "ua_w_k": ua_w_k,
            "ua_ref_w_k": ua_ref_w_k,
            "lmtd_k": lmtd_k,
            "air_m_dot_kg_s": air_m_dot,
            "air_m_dot_nominal_kg_s": air_m_dot_ref,
            "air_cp_j_kg_k": air_cp,
            "air_inlet_c": dock_c,
            "air_outlet_c": air_outlet_c,
            "air_delta_t_k": air_delta_t_k,
            "air_flow_ratio": air_flow_ratio,
            "ua_flow_exponent": ua_exponent,
        }

    def _standalone_air_cycle_constraint_penalty(self, unknowns: np.ndarray) -> np.ndarray:
        defaults_lower = [-83.15, -20.0, -100.0, -100.0, -100.0]
        defaults_upper = [46.85, 120.0, 150.0, 150.0, 150.0]
        sim_cfg = self.cfg.get("simulation", {})
        lower = sim_cfg.get("air_cycle_state_lower_bounds", sim_cfg.get("state_lower_bounds", defaults_lower))
        upper = sim_cfg.get("air_cycle_state_upper_bounds", sim_cfg.get("state_upper_bounds", defaults_upper))
        penalties = []
        for value, lo, hi in zip(unknowns, lower, upper):
            penalties.append(max(0.0, float(lo) - float(value)))
            penalties.append(max(0.0, float(value) - float(hi)))
        penalty_sum = sum(penalties)
        if penalty_sum < 1.0e-7:
            return np.zeros(np.asarray(unknowns, dtype=float).size, dtype=float)
        return np.full(np.asarray(unknowns, dtype=float).size, 1.0e3 + 1.0e6 * penalty_sum, dtype=float)

    def _standalone_transient_regenerator_config(self) -> dict | None:
        air_cfg = self.cfg.get("air_cycle", {})
        reg_cfg = air_cfg.get("regenerator", {})
        if not isinstance(reg_cfg, dict):
            return None

        model = str(reg_cfg.get("model", "")).strip().lower()
        if model not in {"transient_distributed", "transient_lumped", "distributed_transient", "yang_transient"}:
            return None
        return reg_cfg

    def _standalone_transient_regenerator_result(
        self,
        room_k: float,
        t3_k: float,
        solid_c: np.ndarray,
        m_air_kg_s: float,
    ) -> dict[str, float | np.ndarray]:
        reg_cfg = self._standalone_transient_regenerator_config()
        if reg_cfg is None:
            raise ValueError("Transient regenerator configuration is not enabled.")

        solid_k = np.asarray(solid_c, dtype=float) + KELVIN_OFFSET
        cell_count = solid_k.size
        if cell_count <= 0:
            raise ValueError("Transient regenerator requires at least one solid cell.")

        air_cfg = self.cfg["air_cycle"]
        ua_total_w_k = float(reg_cfg.get("ua_w_k", air_cfg["regenerator_ua_w_k"]))
        ua_total_w_k = max(ua_total_w_k, 0.0)
        gas_solid_ua_factor = max(float(reg_cfg.get("gas_solid_ua_factor", 2.0)), 0.0)
        ua_side_cell_w_k = gas_solid_ua_factor * ua_total_w_k / max(cell_count, 1)
        cp_air = max(float(reg_cfg.get("cp_air_j_kg_k", CP_DOCK_AIR)), 1.0e-9)
        capacity_rate_w_k = max(float(m_air_kg_s) * cp_air, 1.0e-9)
        gas_effectiveness = 1.0 - float(np.exp(-ua_side_cell_w_k / capacity_rate_w_k))
        gas_effectiveness = float(np.clip(gas_effectiveness, 0.0, 1.0))

        hot_out_k = np.empty(cell_count, dtype=float)
        cold_out_k = np.empty(cell_count, dtype=float)
        q_hot_to_solid_w = np.zeros(cell_count, dtype=float)
        q_solid_to_cold_w = np.zeros(cell_count, dtype=float)

        hot_in_k = float(t3_k)
        for idx in range(cell_count):
            out_k = hot_in_k + gas_effectiveness * (solid_k[idx] - hot_in_k)
            hot_out_k[idx] = out_k
            q_hot_to_solid_w[idx] = capacity_rate_w_k * (hot_in_k - out_k)
            hot_in_k = out_k

        cold_in_k = float(room_k)
        for idx in range(cell_count - 1, -1, -1):
            out_k = cold_in_k + gas_effectiveness * (solid_k[idx] - cold_in_k)
            cold_out_k[idx] = out_k
            q_solid_to_cold_w[idx] = capacity_rate_w_k * (out_k - cold_in_k)
            cold_in_k = out_k

        ambient_k = float(reg_cfg.get("ambient_k", self.cfg.get("boundary_conditions", {}).get("ambient_c", 30.0) + KELVIN_OFFSET))
        heat_leak_ua_total_w_k = max(float(reg_cfg.get("heat_leak_ua_w_k", 0.0)), 0.0)
        q_leak_to_solid_w = heat_leak_ua_total_w_k * (ambient_k - solid_k) / max(cell_count, 1)
        solid_net_w = q_hot_to_solid_w - q_solid_to_cold_w + q_leak_to_solid_w

        return {
            "t4_k": float(hot_in_k),
            "t6_k": float(cold_in_k),
            "q_reg_hot_w": float(np.sum(q_hot_to_solid_w)),
            "q_reg_cold_w": float(np.sum(q_solid_to_cold_w)),
            "q_reg_solid_net_w": solid_net_w,
            "q_reg_heat_leak_w": float(np.sum(q_leak_to_solid_w)),
            "regenerator_gas_effectiveness": gas_effectiveness,
            "regenerator_solid_mean_k": float(np.mean(solid_k)),
            "regenerator_solid_min_k": float(np.min(solid_k)),
            "regenerator_solid_max_k": float(np.max(solid_k)),
            "regenerator_cell_count": float(cell_count),
            "regenerator_solid_k": solid_k,
            "regenerator_hot_out_k": hot_out_k,
            "regenerator_cold_out_k": cold_out_k,
        }

    def air_cycle_residual(self, unknowns: np.ndarray, prev_state: np.ndarray, time_s: float, dt_s: float) -> np.ndarray:
        unknowns = np.asarray(unknowns, dtype=float)
        room_c, water_loop_c, t3_c, t4_c, t6_c = unknowns[:5]
        prev_room_c, prev_water_loop_c = prev_state[:2]
        room_k = room_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET

        penalty = self._standalone_air_cycle_constraint_penalty(unknowns)
        if np.any(penalty > 0.0):
            return penalty

        caps = self.cfg["thermal_masses"]
        loop_cfg = self._water_loop_config()
        water_capacitance = float(loop_cfg.get("capacitance_j_k", caps.get("water_loop_capacitance_j_k", caps.get("sink_capacitance_j_k", 500000.0))))
        reg_cfg = self._standalone_transient_regenerator_config()

        try:
            air = self._evaluate_air_cycle(room_k, t3_k, t4_k, t6_k)
            hx_uas = self._standalone_air_cycle_uas(air)
            water_cooler = self._evaluate_water_loop_air_cooler(water_loop_c)
            transient_reg = None
            if reg_cfg is not None:
                solid_c = unknowns[5:]
                if solid_c.size <= 0:
                    raise ValueError("Transient regenerator state is missing solid cell temperatures.")
                transient_reg = self._standalone_transient_regenerator_result(room_k, t3_k, solid_c, air["m_air"])
        except ValueError:
            return np.full(unknowns.size, 1.0e9, dtype=float)

        t2_c = air["t2_k"] - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - water_loop_c, t3_c - water_loop_c)
        q_reg_ua = hx_uas["regenerator"]["ua_w_k"] * reg_lmtd
        q_cascade_ua = hx_uas["cascade"]["ua_w_k"] * cascade_lmtd
        room_load_w = self.load_w(time_s) + self._standalone_air_cycle_cold_loss_w(room_k)
        fixed_t3_c = self._fixed_air_cycle_after_cooler_outlet_c(self.cfg["air_cycle"])

        if transient_reg is not None:
            solid_c = unknowns[5:]
            prev_solid_c = np.asarray(prev_state[3:], dtype=float)
            if prev_solid_c.size != solid_c.size:
                prev_solid_c = solid_c.copy()
            solid_cap_total = max(float(reg_cfg.get("solid_capacitance_j_k", 1.0)), 1.0e-9)
            solid_cap_cell = solid_cap_total / max(solid_c.size, 1)
            solid_residuals = solid_c - prev_solid_c - dt_s * np.asarray(transient_reg["q_reg_solid_net_w"], dtype=float) / solid_cap_cell

            common_residuals = [
                room_c - prev_room_c - dt_s * (room_load_w - air["q_room"]) / caps["room_capacitance_j_k"],
            ]
            if fixed_t3_c is not None:
                loop_anchor_c = float(loop_cfg.get("ambient_c", fixed_t3_c))
                common_residuals.extend(
                    [
                        water_loop_c - loop_anchor_c,
                        t4_c - (float(transient_reg["t4_k"]) - KELVIN_OFFSET),
                        t6_c - (float(transient_reg["t6_k"]) - KELVIN_OFFSET),
                        t3_c - fixed_t3_c,
                    ]
                )
            else:
                common_residuals.extend(
                    [
                        water_loop_c - prev_water_loop_c - dt_s * (air["q_cascade"] - water_cooler["q_w"]) / water_capacitance,
                        t4_c - (float(transient_reg["t4_k"]) - KELVIN_OFFSET),
                        t6_c - (float(transient_reg["t6_k"]) - KELVIN_OFFSET),
                        air["q_cascade"] - q_cascade_ua,
                    ]
                )
            return np.concatenate([np.array(common_residuals, dtype=float), solid_residuals])

        if fixed_t3_c is not None:
            loop_anchor_c = float(loop_cfg.get("ambient_c", fixed_t3_c))
            return np.array(
                [
                    room_c - prev_room_c - dt_s * (room_load_w - air["q_room"]) / caps["room_capacitance_j_k"],
                    water_loop_c - loop_anchor_c,
                    air["q_reg_hot"] - air["q_reg_cold"],
                    air["q_reg_hot"] - q_reg_ua,
                    t3_c - fixed_t3_c,
                ],
                dtype=float,
            )

        return np.array(
            [
                room_c - prev_room_c - dt_s * (room_load_w - air["q_room"]) / caps["room_capacitance_j_k"],
                water_loop_c - prev_water_loop_c - dt_s * (air["q_cascade"] - water_cooler["q_w"]) / water_capacitance,
                air["q_reg_hot"] - air["q_reg_cold"],
                air["q_reg_hot"] - q_reg_ua,
                air["q_cascade"] - q_cascade_ua,
            ],
            dtype=float,
        )

    def air_cycle_steady_state_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        steady_state = np.array([unknowns[0], unknowns[1]], dtype=float)
        return self.air_cycle_residual(unknowns, steady_state, time_s, dt_s=1.0)

    def air_cycle_post_process(self, unknowns: np.ndarray, time_s: float) -> StepResult:
        unknowns = np.asarray(unknowns, dtype=float)
        room_c, water_loop_c, t3_c, t4_c, t6_c = unknowns[:5]
        room_k = room_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        air_cfg = self.cfg["air_cycle"]
        self._update_room_moisture_properties(room_k, air_cfg["p_low_pa"])
        air = self._evaluate_air_cycle(room_k, t3_k, t4_k, t6_k)
        hx_uas = self._standalone_air_cycle_uas(air)
        water_cooler = self._evaluate_water_loop_air_cooler(water_loop_c)
        infiltration = self.infiltration_disturbance_w(time_s)
        air_input_power = self._air_cycle_input_power(air, air_cfg)
        cop = air["q_room"] / max(air_input_power, 1.0)
        t2_c = air["t2_k"] - KELVIN_OFFSET
        t5_c = air["t5_k"] - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - water_loop_c, t3_c - water_loop_c)
        reg_cfg = self._standalone_transient_regenerator_config()
        transient_reg = None
        if reg_cfg is not None and unknowns.size > 5:
            transient_reg = self._standalone_transient_regenerator_result(room_k, t3_k, unknowns[5:], air["m_air"])

        refrigerating_source = str(air_cfg.get("refrigerating_temperature_source", "room")).strip().lower()
        if refrigerating_source in {"supply", "t5", "air_entering_room"}:
            refrigerating_temperature_k = air["t5_k"]
        elif refrigerating_source in {"return", "t6", "regenerator_cold_outlet"}:
            refrigerating_temperature_k = t6_k
        elif refrigerating_source in {"expander_inlet", "t4"}:
            refrigerating_temperature_k = t4_k
        else:
            refrigerating_temperature_k = room_k
        cold_loss_w = self._standalone_air_cycle_cold_loss_w(room_k)
        room_load_w = self.load_w(time_s) + cold_loss_w

        values = {
            "time_s": time_s,
            "room_c": room_c,
            "room_k": room_k,
            "refrigerating_temperature_c": refrigerating_temperature_k - KELVIN_OFFSET,
            "refrigerating_temperature_k": refrigerating_temperature_k,
            "water_loop_c": water_loop_c,
            "water_loop_k": water_loop_c + KELVIN_OFFSET,
            "sink_c": water_loop_c,
            "sink_k": water_loop_c + KELVIN_OFFSET,
            "dock_c": float("nan"),
            "dock_k": float("nan"),
            "t2_c": t2_c,
            "t2_k": air["t2_k"],
            "t3_c": t3_c,
            "t3_k": t3_k,
            "t4_c": t4_c,
            "t4_k": t4_k,
            "t5_c": t5_c,
            "t5_k": air["t5_k"],
            "t6_c": t6_c,
            "t6_k": t6_k,
            "q_room_w": air["q_room"],
            "q_useful_w": air["q_room"],
            "q_cascade_w": air["q_cascade"],
            "q_cascade_ua_w": hx_uas["cascade"]["ua_w_k"] * cascade_lmtd,
            "q_water_loop_air_cooler_w": water_cooler["q_w"],
            "q_cond_w": water_cooler["q_w"],
            "w_air_comp_w": air["w_air_comp"],
            "w_air_turb_w": air["w_air_turb"],
            "w_air_input_w": air_input_power,
            "w_total_input_w": air_input_power,
            "cop_air_cycle": cop,
            "cop_system": cop,
            "m_air_kg_s": air["m_air"],
            "m_ref_kg_s": 0.0,
            "air_low_pressure_pa": air["p1"],
            "air_high_pressure_pa": air["p2"],
            "air_pressure_ratio": air["pressure_ratio"],
            "air_damper_opening": air["damper_opening"],
            "air_damper_resistance_head_coefficient": air["damper_resistance_head_coefficient"],
            "air_compressor_map_mass_flow_kg_s": air["m_air_map"],
            "air_compressor_suction_density_kg_m3": air["rho1"],
            "air_compressor_suction_mixture_density_kg_m3": air["rho1_mixture"],
            "air_compressor_volumetric_flow_m3_s": air["m_air"] / max(air["rho1"], 1.0e-9),
            "air_compressor_actual_head_j_kg": air["compressor_head_actual"],
            "air_compressor_actual_head_m": air["compressor_head_actual"] / 9.80665,
            "air_compressor_isentropic_head_j_kg": air["compressor_head_is"],
            "air_compressor_isentropic_head_m": air["compressor_head_is"] / 9.80665,
            "air_compressor_eta_is_target": air["compressor_eta_is_target"],
            "air_compressor_eta_is_effective": air["compressor_eta_is_effective"],
            "regenerator_ua_w_k": hx_uas["regenerator"]["ua_w_k"],
            "regenerator_ua_ref_w_k": hx_uas["regenerator"]["ua_ref_w_k"],
            "regenerator_air_flow_ratio": hx_uas["regenerator"]["flow_ratio"],
            "regenerator_ua_flow_exponent": hx_uas["regenerator"]["exponent"],
            "regenerator_lmtd_k": reg_lmtd,
            "cascade_ua_w_k": hx_uas["cascade"]["ua_w_k"],
            "cascade_ua_ref_w_k": hx_uas["cascade"]["ua_ref_w_k"],
            "cascade_air_flow_ratio": hx_uas["cascade"]["flow_ratio"],
            "cascade_ua_flow_exponent": hx_uas["cascade"]["exponent"],
            "cascade_lmtd_k": cascade_lmtd,
            "water_loop_air_cooler_ua_w_k": water_cooler["ua_w_k"],
            "water_loop_air_cooler_ua_ref_w_k": water_cooler["ua_ref_w_k"],
            "water_loop_air_cooler_air_flow_ratio": water_cooler["flow_ratio"],
            "water_loop_air_cooler_ua_flow_exponent": water_cooler["ua_flow_exponent"],
            "water_loop_air_cooler_ambient_c": water_cooler["ambient_c"],
            "water_loop_air_cooler_delta_t_k": water_cooler["delta_t_k"],
            "humidity_ratio_room_kg_kg_da": air["humidity_ratio_room"],
            "room_relative_humidity": self._room_moisture.relative_humidity,
            "room_saturation_humidity_ratio_kg_kg_da": self._room_moisture.saturation_humidity_ratio,
            "room_dry_air_mass_kg": self._room_moisture.dry_air_mass_kg,
            "room_moisture_infiltration_dry_air_mass_flow_kg_s": self._room_moisture.infiltration_dry_air_mass_flow_kg_s,
            "room_moisture_deposition_rate_kg_s": self._room_moisture.deposition_rate_kg_s,
            "room_moisture_deposition_load_w": self._room_moisture.q_deposition_w,
            "room_moisture_cumulative_deposited_water_kg": self._room_moisture.cumulative_deposited_water_kg,
            "humidity_ratio_supply_vapor_kg_kg_da": air["humidity_ratio_5_vapor"],
            "humidity_ratio_supply_ice_kg_kg_da": air["humidity_ratio_5_ice"],
            "ice_mass_flow_kg_s": air["ice_mass_flow"],
            "base_room_load_w": self._base_room_load_w(time_s),
            "air_cycle_cold_loss_w": cold_loss_w,
            "infiltration_room_w": infiltration["room_w"],
            "infiltration_q_m3_s": infiltration.get("q_m3_s", 0.0),
            "infiltration_sensible_w": infiltration.get("q_sensible_w", 0.0),
            "infiltration_latent_w": infiltration.get("q_latent_w", 0.0),
            "infiltration_total_w": infiltration.get("q_total_w", infiltration["room_w"]),
            "infiltration_cumulative_volume_m3": infiltration.get("cumulative_volume_m3", 0.0),
            "infiltration_stage": infiltration.get("stage", 0.0),
            "infiltration_door_open_fraction": infiltration.get("door_open_fraction", 0.0),
            "load_w": room_load_w,
        }
        if transient_reg is not None:
            values.update(
                {
                    "regenerator_model_transient": 1.0,
                    "regenerator_cell_count": transient_reg["regenerator_cell_count"],
                    "regenerator_gas_effectiveness": transient_reg["regenerator_gas_effectiveness"],
                    "regenerator_solid_mean_k": transient_reg["regenerator_solid_mean_k"],
                    "regenerator_solid_min_k": transient_reg["regenerator_solid_min_k"],
                    "regenerator_solid_max_k": transient_reg["regenerator_solid_max_k"],
                    "regenerator_q_hot_to_solid_w": transient_reg["q_reg_hot_w"],
                    "regenerator_q_solid_to_cold_w": transient_reg["q_reg_cold_w"],
                    "regenerator_q_solid_net_total_w": float(np.sum(np.asarray(transient_reg["q_reg_solid_net_w"], dtype=float))),
                    "regenerator_q_heat_leak_w": transient_reg["q_reg_heat_leak_w"],
                }
            )
        else:
            values["regenerator_model_transient"] = 0.0

        state_values = [room_c, water_loop_c, self._room_moisture.humidity_ratio]
        if reg_cfg is not None and unknowns.size > 5:
            state_values.extend(float(value) for value in unknowns[5:])
        return StepResult(
            values=values,
            state_vector=np.array(state_values, dtype=float),
        )

    def _vcc_constraint_penalty(self, unknowns: np.ndarray) -> np.ndarray:
        sink_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock = unknowns[:5]
        penalties = [
            max(0.0, -3.15 - sink_c),
            max(0.0, sink_c - 86.85),
            max(0.0, -73.15 - tevap_c),
            max(0.0, tevap_c - 46.85),
            max(0.0, 0.0 - tcond_c),
            max(0.0, tcond_c - 90.0),
            max(0.0, 1.0e-6 - m_ref_cascade),
            max(0.0, 1.0e-6 - m_ref_dock),
            max(0.0, self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock) - 5.0),
        ]
        residual_size = 5 + self._receiver_extra_residual_count()
        if self.high_pressure_receiver_enabled():
            receiver_mass = self._vcc_receiver_unknown_mass(unknowns)
            receiver_cfg = self._receiver_config()
            receiver_mass_max = float(receiver_cfg.get("mass_max_kg", receiver_cfg.get("maximum_mass_kg", 100.0)))
            penalties.extend(
                [
                    max(0.0, float(receiver_cfg.get("mass_min_kg", 0.0)) - receiver_mass),
                    max(0.0, receiver_mass - receiver_mass_max),
                ]
            )
        if self.lpr_subcooling_control_enabled():
            subcooling_k = self._lpr_subcooling_unknown(unknowns, 5)
            lpr_cfg = self._low_pressure_receiver_config()
            penalties.extend(
                [
                    max(0.0, float(lpr_cfg.get("subcooling_min_k", 0.0)) - subcooling_k),
                    max(0.0, subcooling_k - float(lpr_cfg.get("subcooling_max_k", 30.0))),
                ]
            )
        penalty_sum = sum(penalties)
        if penalty_sum < 1.0e-7:
            return np.zeros(residual_size, dtype=float)
        return np.full(residual_size, 1.0e3 + 1.0e6 * penalty_sum, dtype=float)

    def vcc_residual(self, unknowns: np.ndarray, prev_state: np.ndarray, time_s: float, dt_s: float) -> np.ndarray:
        sink_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock = unknowns[:5]
        receiver_mass_kg = self._vcc_receiver_unknown_mass(unknowns)
        subcooling_k = self._lpr_subcooling_unknown(unknowns, 5)
        m_ref = self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock)
        prev_sink_c = prev_state[0]
        prev_receiver_mass_kg = prev_state[1] if self.high_pressure_receiver_enabled() and len(prev_state) > 1 else receiver_mass_kg
        sink_k = sink_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET
        tcond_k = tcond_c + KELVIN_OFFSET

        penalty = self._vcc_constraint_penalty(unknowns)
        if np.any(penalty > 0.0):
            return penalty

        bc = self.cfg["boundary_conditions"]
        caps = self.cfg["thermal_masses"]
        ambient_c = bc["ambient_c"]
        loads = self._vcc_standalone_loads_w(time_s)
        try:
            ref = self._evaluate_refrigerant_cycle(
                tevap_k,
                tcond_k,
                m_ref_cascade,
                m_ref_dock,
                loads["cascade_w"],
                loads["dock_w"],
                receiver_mass_kg,
                subcooling_k,
            )
            condenser_ua = self._vcc_condenser_ua_result()
        except ValueError:
            size = 5 + self._receiver_extra_residual_count()
            if compressor_uses_map_cooling_capacity_balance(self.cfg["vcc_cycle"]["compressor"]):
                size += 1
            return np.full(size, 1.0e9, dtype=float)

        condenser_lmtd = positive_lmtd(tcond_c - ambient_c, tcond_c - sink_c)
        q_cond_ua = condenser_ua["ua_w_k"] * condenser_lmtd
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - ambient_c)
        residuals = [
            sink_c - prev_sink_c - dt_s * (ref["q_cond"] - sink_rejection) / caps["sink_capacitance_j_k"],
            ref["q_cond"] - q_cond_ua,
            m_ref_cascade - ref["m_ref_valve_cascade"],
            m_ref_dock - ref["m_ref_valve_dock"],
            m_ref - ref["m_ref_compressor"],
        ]
        if self.high_pressure_receiver_enabled():
            residuals.append(self._receiver_mass_balance_residual(receiver_mass_kg, prev_receiver_mass_kg, ref, dt_s))
        if self.lpr_subcooling_control_enabled():
            prev_idx = 1 + (1 if self.high_pressure_receiver_enabled() else 0)
            prev_subcooling_k = prev_state[prev_idx] if len(prev_state) > prev_idx else subcooling_k
            residuals.append(self._lpr_subcooling_balance_residual(subcooling_k, prev_subcooling_k, ref, dt_s))
        if ref["uses_map_cooling_capacity_balance"]:
            residuals.append(ref["q_map_capacity_balance_error"])
        return np.asarray(residuals, dtype=float)

    def vcc_steady_state_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        steady_state = [float(unknowns[0])]
        if self.high_pressure_receiver_enabled() and len(unknowns) > 5:
            steady_state.append(float(unknowns[5]))
        if self.lpr_subcooling_control_enabled():
            idx = 5 + (1 if self.high_pressure_receiver_enabled() else 0)
            steady_state.append(float(unknowns[idx]))
        steady_state = np.array(steady_state, dtype=float)
        return self.vcc_residual(unknowns, steady_state, time_s, dt_s=1.0)

    def vcc_post_process(self, unknowns: np.ndarray, time_s: float) -> StepResult:
        sink_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock = unknowns[:5]
        receiver_mass_kg = self._vcc_receiver_unknown_mass(unknowns)
        subcooling_k = self._lpr_subcooling_unknown(unknowns, 5)
        m_ref = self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock)
        tevap_k = tevap_c + KELVIN_OFFSET
        tcond_k = tcond_c + KELVIN_OFFSET
        bc = self.cfg["boundary_conditions"]
        loads = self._vcc_standalone_loads_w(time_s)
        ref = self._evaluate_refrigerant_cycle(
            tevap_k,
            tcond_k,
            m_ref_cascade,
            m_ref_dock,
            loads["cascade_w"],
            loads["dock_w"],
            receiver_mass_kg,
            subcooling_k,
        )
        branch_holdup = self._branch_holdup_outputs(ref, tevap_k, time_s)
        condenser_ua = self._vcc_condenser_ua_result()
        condenser_lmtd = positive_lmtd(tcond_c - bc["ambient_c"], tcond_c - sink_c)
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - bc["ambient_c"])
        cop = ref["q_evap_total"] / max(ref["w_ref_comp"], 1.0)
        values = {
            "time_s": time_s,
            "room_c": float("nan"),
            "dock_c": float("nan"),
            "sink_c": sink_c,
            "tevap_c": tevap_c,
            "tcond_c": tcond_c,
            "t7_c": ref["t7_k"] - KELVIN_OFFSET,
            "t8_c": ref["t8_k"] - KELVIN_OFFSET,
            "t2_c": float("nan"),
            "t3_c": float("nan"),
            "t4_c": float("nan"),
            "t5_c": float("nan"),
            "t6_c": float("nan"),
            "q_room_w": 0.0,
            "q_dock_w": loads["dock_w"],
            "q_useful_w": ref["q_evap_total"],
            "q_cascade_w": loads["cascade_w"],
            "q_cascade_refrigerant_w": ref["q_cascade_branch"],
            "q_dock_refrigerant_w": ref["q_dock_branch"],
            "q_evap_total_w": ref["q_evap_total"],
            "q_evap_load_requested_w": ref["q_evap_load_requested"],
            "q_map_capacity_balance_error_w": ref["q_map_capacity_balance_error"],
            "q_cond_w": ref["q_cond"],
            "q_condenser_ua_w": condenser_ua["ua_w_k"] * condenser_lmtd,
            "q_sink_rejection_w": sink_rejection,
            "w_ref_comp_w": ref["w_ref_comp"],
            "w_total_input_w": ref["w_ref_comp"],
            "w_ref_isentropic_w": ref["w_ref_isentropic"],
            "cop_vcc": cop,
            "cop_system": cop,
            "m_ref_kg_s": m_ref,
            "m_air_kg_s": 0.0,
            "condenser_ua_w_k": condenser_ua["ua_w_k"],
            "condenser_ua_ref_w_k": condenser_ua["ua_ref_w_k"],
            "condenser_sink_flow_ratio": condenser_ua["flow_ratio"],
            "condenser_ua_flow_exponent": condenser_ua["exponent"],
            "condenser_lmtd_k": condenser_lmtd,
            "refrigerant_evaporating_pressure_pa": ref["p_evap"],
            "refrigerant_condensing_pressure_pa": ref["p_cond"],
            "refrigerant_pressure_ratio": ref["pressure_ratio"],
            "refrigerant_condensing_pressure_from_tcond_pa": ref["p_cond_from_tcond"],
            "refrigerant_pressure_ratio_from_tcond": ref["pressure_ratio_from_tcond"],
            "refrigerant_compressor_eta_is_effective": ref["eta_is_effective"],
            "refrigerant_compressor_speed_rpm": ref["compressor_speed_rpm"],
            "refrigerant_compressor_map_capacity_w": ref["compressor_map_q_w"],
            "refrigerant_compressor_volumetric_efficiency": ref["compressor_eta_v"],
            "refrigerant_compressor_suction_density_kg_m3": ref["compressor_suction_density_kg_m3"],
            "refrigerant_compressor_displacement_m3_per_rev": ref["compressor_displacement_m3_per_rev"],
            "low_pressure_receiver_enabled": ref["low_pressure_receiver_enabled"],
            "refrigerant_evaporators_series": ref["evaporator_arrangement_series"],
            "refrigerant_evaporator_outlet_c": ref["t7_evaporator_out_k"] - KELVIN_OFFSET,
            "valve_flow_kg_s": ref["m_ref_valve"],
            "cascade_valve_flow_kg_s": ref["m_ref_valve_cascade"],
            "dock_valve_flow_kg_s": ref["m_ref_valve_dock"],
            "compressor_map_flow_kg_s": ref["m_ref_compressor"],
            "receiver_enabled": ref["receiver_enabled"],
            "receiver_mass_kg": ref["receiver_mass_kg"],
            "receiver_volume_m3": ref["receiver_volume_m3"],
            "receiver_liquid_mass_kg": ref["receiver_liquid_mass_kg"],
            "receiver_vapor_mass_kg": ref["receiver_vapor_mass_kg"],
            "receiver_liquid_volume_m3": ref["receiver_liquid_volume_m3"],
            "receiver_liquid_fill_fraction": ref["receiver_liquid_fill_fraction"],
            "receiver_liquid_fill_fraction_raw": ref["receiver_liquid_fill_fraction_raw"],
            "receiver_overfill_mass_kg": ref["receiver_overfill_mass_kg"],
            "receiver_liquid_feed_fraction": ref["receiver_liquid_feed_fraction"],
            "receiver_starvation_flow_multiplier": ref["receiver_starvation_flow_multiplier"],
            "receiver_outlet_enthalpy_j_kg": ref["receiver_outlet_enthalpy_j_kg"],
            "receiver_outlet_density_kg_m3": ref["receiver_outlet_density_kg_m3"],
            "receiver_inlet_m_dot_kg_s": ref["receiver_inlet_m_dot_kg_s"],
            "receiver_outlet_m_dot_kg_s": ref["m_ref_valve"],
            "m_ref_cascade_kg_s": ref["m_ref_cascade"],
            "m_ref_dock_kg_s": ref["m_ref_dock"],
            "cascade_valve_opening": ref["cascade_valve_opening"],
            "dock_valve_opening": ref["dock_valve_opening"],
            "refrigerant_subcooling_k": ref["subcooling_k"],
            "refrigerant_superheat_k": branch_holdup["superheat_k"],
            "refrigerant_cascade_superheat_k": branch_holdup["superheat_cascade_k"],
            "refrigerant_dock_superheat_k": branch_holdup["superheat_dock_k"],
            "refrigerant_raw_superheat_k": ref["superheat_k"],
            "refrigerant_raw_cascade_superheat_k": ref["superheat_cascade_k"],
            "refrigerant_raw_dock_superheat_k": ref["superheat_dock_k"],
            "refrigerant_compressor_work_w": ref["w_ref_comp"],
            "refrigerant_compressor_isentropic_work_w": ref["w_ref_isentropic"],
            "refrigerant_q_condenser_stage_w": ref["q_condenser_stage"],
            "refrigerant_q_subcooler_w": ref["q_subcooler"],
            "refrigerant_q_oil_cooling_w": ref["q_oil_cooling_w"],
            "refrigerant_compressor_eta_is_target": ref["eta_is_target"],
            "base_vcc_cascade_load_w": loads["cascade_w"],
            "base_vcc_dock_load_w": loads["dock_w"],
            "load_w": loads["total_w"],
        }
        state_vector = [sink_c]
        if self.high_pressure_receiver_enabled():
            state_vector.append(receiver_mass_kg)
        if self.lpr_subcooling_control_enabled():
            state_vector.append(subcooling_k)
        return StepResult(values=values, state_vector=np.array(state_vector, dtype=float))

    def residual(self, unknowns: np.ndarray, prev_state: np.ndarray, time_s: float, dt_s: float) -> np.ndarray:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock, dock_c = unknowns[:10]
        receiver_mass_kg = self._receiver_unknown_mass(unknowns)
        subcooling_k = self._lpr_subcooling_unknown(unknowns, 10)
        m_ref = self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock)
        prev_room_c, prev_sink_c, prev_dock_c = prev_state[:3]
        room_k = room_c + KELVIN_OFFSET
        sink_k = sink_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET
        tcond_k = tcond_c + KELVIN_OFFSET

        penalty = self._constraint_penalty(unknowns)
        if np.any(penalty > 0.0):
            return penalty

        air_cfg = self.cfg["air_cycle"]
        vcc_cfg = self.cfg["vcc_cycle"]
        bc = self.cfg["boundary_conditions"]
        caps = self.cfg["thermal_masses"]
        ambient_c = bc["ambient_c"]

        try:
            air = self._evaluate_air_cycle(room_k, t3_k, t4_k, t6_k)
            dock_evap = self._evaluate_dock_evaporator(dock_c, tevap_c)
            q_dock_evap = dock_evap["q_w"]
            ref = self._evaluate_refrigerant_cycle(
                tevap_k,
                tcond_k,
                m_ref_cascade,
                m_ref_dock,
                air["q_cascade"],
                q_dock_evap,
                receiver_mass_kg,
                subcooling_k,
            )
            hx_uas = self._heat_exchanger_uas(air)
        except ValueError:
            size = 10 + self._receiver_extra_residual_count()
            return np.full(size, 1.0e9, dtype=float)

        t2_c = air["t2_k"] - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - tevap_c, t3_c - tevap_c)
        condenser_lmtd = positive_lmtd(tcond_c - ambient_c, tcond_c - sink_c)

        q_reg_ua = hx_uas["regenerator"]["ua_w_k"] * reg_lmtd
        q_cascade_ua = hx_uas["cascade"]["ua_w_k"] * cascade_lmtd
        q_cond_ua = hx_uas["condenser"]["ua_w_k"] * condenser_lmtd
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - ambient_c)
        room_load_w = self.load_w(time_s)
        dock_load_w = self.dock_load_w(time_s)
        if ref["uses_map_cooling_capacity_balance"]:
            cascade_balance = ref["q_evap_load_requested"] - ref["compressor_map_q_w"]
        else:
            cascade_balance = air["q_cascade"] - q_cascade_ua

        residuals = [
            room_c - prev_room_c - dt_s * (room_load_w - air["q_room"]) / caps["room_capacitance_j_k"],
            dock_c - prev_dock_c - dt_s * (dock_load_w - q_dock_evap) / caps["dock_capacitance_j_k"],
            sink_c - prev_sink_c - dt_s * (ref["q_cond"] - sink_rejection) / caps["sink_capacitance_j_k"],
            air["q_reg_hot"] - air["q_reg_cold"],
            air["q_reg_hot"] - q_reg_ua,
            cascade_balance,
            ref["q_cond"] - q_cond_ua,
            m_ref_cascade - ref["m_ref_valve_cascade"],
            m_ref_dock - ref["m_ref_valve_dock"],
            m_ref - ref["m_ref_compressor"],
        ]
        if self.high_pressure_receiver_enabled():
            prev_receiver_mass_kg = prev_state[4] if len(prev_state) > 4 else receiver_mass_kg
            residuals.append(self._receiver_mass_balance_residual(receiver_mass_kg, prev_receiver_mass_kg, ref, dt_s))
        if self.lpr_subcooling_control_enabled():
            prev_idx = 4 + (1 if self.high_pressure_receiver_enabled() else 0)
            prev_subcooling_k = prev_state[prev_idx] if len(prev_state) > prev_idx else subcooling_k
            residuals.append(self._lpr_subcooling_balance_residual(subcooling_k, prev_subcooling_k, ref, dt_s))
        return np.asarray(residuals, dtype=float)

    def steady_state_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        steady_state = [float(unknowns[0]), float(unknowns[1]), float(unknowns[9]), 0.0]
        if self.high_pressure_receiver_enabled() and len(unknowns) > 10:
            steady_state.append(float(unknowns[10]))
        if self.lpr_subcooling_control_enabled():
            idx = 10 + (1 if self.high_pressure_receiver_enabled() else 0)
            steady_state.append(float(unknowns[idx]))
        steady_state = np.array(steady_state, dtype=float)
        return self.residual(unknowns, steady_state, time_s, dt_s=1.0)

    def startup_balance_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        balances, _ = self.startup_evaluation(unknowns, time_s)
        return balances

    def startup_metrics(self, unknowns: np.ndarray, time_s: float) -> dict[str, float]:
        _, metrics = self.startup_evaluation(unknowns, time_s)
        return metrics

    def post_process(self, unknowns: np.ndarray, time_s: float) -> StepResult:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref_cascade, m_ref_dock, dock_c = unknowns[:10]
        receiver_mass_kg = self._receiver_unknown_mass(unknowns)
        subcooling_k = self._lpr_subcooling_unknown(unknowns, 10)
        m_ref = self.effective_refrigerant_mass_flow(m_ref_cascade, m_ref_dock)
        room_k = room_c + KELVIN_OFFSET
        sink_k = sink_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET
        tcond_k = tcond_c + KELVIN_OFFSET
        air_cfg = self.cfg["air_cycle"]
        self._update_room_moisture_properties(room_k, air_cfg["p_low_pa"])
        air = self._evaluate_air_cycle(room_k, t3_k, t4_k, t6_k)
        hx_uas = self._heat_exchanger_uas(air)
        dock_evap = self._evaluate_dock_evaporator(dock_c, tevap_c)
        q_dock = dock_evap["q_w"]
        infiltration = self.infiltration_disturbance_w(time_s)
        ref = self._evaluate_refrigerant_cycle(
            tevap_k,
            tcond_k,
            m_ref_cascade,
            m_ref_dock,
            air["q_cascade"],
            q_dock,
            receiver_mass_kg,
            subcooling_k,
        )
        branch_holdup = self._branch_holdup_outputs(ref, tevap_k, time_s)
        air_input_power = self._air_cycle_input_power(air, air_cfg)
        q_dock_external = q_dock
        useful_cooling = air["q_room"] + q_dock_external
        total_input_power = air_input_power + ref["w_ref_comp"]
        cop = useful_cooling / max(total_input_power, 1.0)
        cop_room_only = air["q_room"] / max(total_input_power, 1.0)
        t2_c = air["t2_k"] - KELVIN_OFFSET
        t5_c = air["t5_k"] - KELVIN_OFFSET
        t7_c = ref["t7_k"] - KELVIN_OFFSET
        infiltration_cfg = self.cfg.get("disturbances", {}).get("infiltration", {})

        values = {
            "time_s": time_s,
            "room_c": room_c,
            "dock_c": dock_c,
            "sink_c": sink_c,
            "t2_c": t2_c,
            "t3_c": t3_c,
            "t4_c": t4_c,
            "t5_c": t5_c,
            "t6_c": t6_c,
            "t7_c": t7_c,
            "tevap_c": tevap_c,
            "tcond_c": tcond_c,
            "q_room_w": air["q_room"],
            "q_dock_w": q_dock,
            "q_dock_external_w": q_dock_external,
            "q_useful_w": useful_cooling,
            "q_cascade_w": air["q_cascade"],
            "q_cascade_refrigerant_w": ref["q_cascade_branch"],
            "q_dock_refrigerant_w": ref["q_dock_branch"],
            "q_evap_total_w": ref["q_evap_total"],
            "q_evap_load_requested_w": ref["q_evap_load_requested"],
            "q_map_capacity_balance_error_w": ref["q_map_capacity_balance_error"],
            "q_cond_w": ref["q_cond"],
            "refrigerant_q_condenser_stage_w": ref["q_condenser_stage"],
            "refrigerant_q_subcooler_w": ref["q_subcooler"],
            "refrigerant_q_oil_cooling_w": ref["q_oil_cooling_w"],
            "w_air_comp_w": air["w_air_comp"],
            "w_air_turb_w": air["w_air_turb"],
            "w_air_input_w": air_input_power,
            "w_ref_comp_w": ref["w_ref_comp"],
            "w_total_input_w": total_input_power,
            "w_ref_isentropic_w": ref["w_ref_isentropic"],
            "refrigerant_compressor_speed_rpm": ref["compressor_speed_rpm"],
            "refrigerant_compressor_map_capacity_w": ref["compressor_map_q_w"],
            "refrigerant_compressor_volumetric_efficiency": ref["compressor_eta_v"],
            "refrigerant_compressor_suction_density_kg_m3": ref["compressor_suction_density_kg_m3"],
            "refrigerant_compressor_displacement_m3_per_rev": ref["compressor_displacement_m3_per_rev"],
            "low_pressure_receiver_enabled": ref["low_pressure_receiver_enabled"],
            "refrigerant_evaporators_series": ref["evaporator_arrangement_series"],
            "refrigerant_evaporator_outlet_c": ref["t7_evaporator_out_k"] - KELVIN_OFFSET,
            "refrigerant_evaporating_pressure_pa": ref["p_evap"],
            "refrigerant_condensing_pressure_pa": ref["p_cond"],
            "refrigerant_pressure_ratio": ref["pressure_ratio"],
            "refrigerant_condensing_pressure_from_tcond_pa": ref["p_cond_from_tcond"],
            "refrigerant_pressure_ratio_from_tcond": ref["pressure_ratio_from_tcond"],
            "refrigerant_compressor_eta_is_effective": ref["eta_is_effective"],
            "m_ref_kg_s": m_ref,
            "m_air_kg_s": air["m_air"],
            "air_low_pressure_pa": air["p1"],
            "air_high_pressure_pa": air["p2"],
            "air_pressure_ratio": air["pressure_ratio"],
            "air_damper_opening": air["damper_opening"],
            "air_damper_resistance_head_coefficient": air["damper_resistance_head_coefficient"],
            "air_compressor_map_mass_flow_kg_s": air["m_air_map"],
            "air_compressor_suction_density_kg_m3": air["rho1"],
            "air_compressor_suction_mixture_density_kg_m3": air["rho1_mixture"],
            "air_compressor_volumetric_flow_m3_s": air["m_air"] / max(air["rho1"], 1.0e-9),
            "air_compressor_actual_head_j_kg": air["compressor_head_actual"],
            "air_compressor_actual_head_m": air["compressor_head_actual"] / 9.80665,
            "air_compressor_isentropic_head_j_kg": air["compressor_head_is"],
            "air_compressor_isentropic_head_m": air["compressor_head_is"] / 9.80665,
            "air_compressor_eta_is_target": air["compressor_eta_is_target"],
            "air_compressor_eta_is_effective": air["compressor_eta_is_effective"],
            "regenerator_ua_w_k": hx_uas["regenerator"]["ua_w_k"],
            "regenerator_ua_ref_w_k": hx_uas["regenerator"]["ua_ref_w_k"],
            "regenerator_air_flow_ratio": hx_uas["regenerator"]["flow_ratio"],
            "regenerator_ua_flow_exponent": hx_uas["regenerator"]["exponent"],
            "cascade_ua_w_k": hx_uas["cascade"]["ua_w_k"],
            "cascade_ua_ref_w_k": hx_uas["cascade"]["ua_ref_w_k"],
            "cascade_air_flow_ratio": hx_uas["cascade"]["flow_ratio"],
            "cascade_ua_flow_exponent": hx_uas["cascade"]["exponent"],
            "condenser_ua_w_k": hx_uas["condenser"]["ua_w_k"],
            "condenser_ua_ref_w_k": hx_uas["condenser"]["ua_ref_w_k"],
            "condenser_sink_flow_ratio": hx_uas["condenser"]["flow_ratio"],
            "condenser_ua_flow_exponent": hx_uas["condenser"]["exponent"],
            "dock_evaporator_ua_w_k": dock_evap["ua_w_k"],
            "dock_evaporator_ua_ref_w_k": dock_evap["ua_ref_w_k"],
            "dock_evaporator_lmtd_k": dock_evap["lmtd_k"],
            "dock_evaporator_air_inlet_c": dock_evap["air_inlet_c"],
            "dock_evaporator_air_outlet_c": dock_evap["air_outlet_c"],
            "dock_evaporator_air_delta_t_k": dock_evap["air_delta_t_k"],
            "dock_air_m_dot_kg_s": dock_evap["air_m_dot_kg_s"],
            "dock_air_m_dot_nominal_kg_s": dock_evap["air_m_dot_nominal_kg_s"],
            "dock_air_flow_ratio": dock_evap["air_flow_ratio"],
            "dock_evaporator_ua_flow_exponent": dock_evap["ua_flow_exponent"],
            "humidity_ratio_room_kg_kg_da": air["humidity_ratio_room"],
            "room_relative_humidity": self._room_moisture.relative_humidity,
            "room_saturation_humidity_ratio_kg_kg_da": self._room_moisture.saturation_humidity_ratio,
            "room_dry_air_mass_kg": self._room_moisture.dry_air_mass_kg,
            "room_moisture_infiltration_dry_air_mass_flow_kg_s": self._room_moisture.infiltration_dry_air_mass_flow_kg_s,
            "room_moisture_deposition_rate_kg_s": self._room_moisture.deposition_rate_kg_s,
            "room_moisture_deposition_load_w": self._room_moisture.q_deposition_w,
            "room_moisture_cumulative_deposited_water_kg": self._room_moisture.cumulative_deposited_water_kg,
            "humidity_ratio_supply_vapor_kg_kg_da": air["humidity_ratio_5_vapor"],
            "humidity_ratio_supply_ice_kg_kg_da": air["humidity_ratio_5_ice"],
            "ice_mass_flow_kg_s": air["ice_mass_flow"],
            "valve_flow_kg_s": ref["m_ref_valve"],
            "cascade_valve_flow_kg_s": ref["m_ref_valve_cascade"],
            "dock_valve_flow_kg_s": ref["m_ref_valve_dock"],
            "compressor_map_flow_kg_s": ref["m_ref_compressor"],
            "receiver_enabled": ref["receiver_enabled"],
            "receiver_mass_kg": ref["receiver_mass_kg"],
            "receiver_volume_m3": ref["receiver_volume_m3"],
            "receiver_liquid_mass_kg": ref["receiver_liquid_mass_kg"],
            "receiver_vapor_mass_kg": ref["receiver_vapor_mass_kg"],
            "receiver_liquid_volume_m3": ref["receiver_liquid_volume_m3"],
            "receiver_liquid_fill_fraction": ref["receiver_liquid_fill_fraction"],
            "receiver_liquid_fill_fraction_raw": ref["receiver_liquid_fill_fraction_raw"],
            "receiver_overfill_mass_kg": ref["receiver_overfill_mass_kg"],
            "receiver_liquid_feed_fraction": ref["receiver_liquid_feed_fraction"],
            "receiver_starvation_flow_multiplier": ref["receiver_starvation_flow_multiplier"],
            "receiver_outlet_enthalpy_j_kg": ref["receiver_outlet_enthalpy_j_kg"],
            "receiver_outlet_density_kg_m3": ref["receiver_outlet_density_kg_m3"],
            "receiver_inlet_m_dot_kg_s": ref["receiver_inlet_m_dot_kg_s"],
            "receiver_outlet_m_dot_kg_s": ref["m_ref_valve"],
            "m_ref_cascade_kg_s": ref["m_ref_cascade"],
            "m_ref_dock_kg_s": ref["m_ref_dock"],
            "valve_opening": self.cfg["vcc_cycle"].get("expansion_valve", {}).get("opening", ref["cascade_valve_opening"]),
            "cascade_valve_opening": ref["cascade_valve_opening"],
            "dock_valve_opening": ref["dock_valve_opening"],
            "refrigerant_subcooling_k": ref["subcooling_k"],
            "refrigerant_superheat_k": branch_holdup["superheat_k"],
            "refrigerant_cascade_superheat_k": branch_holdup["superheat_cascade_k"],
            "refrigerant_dock_superheat_k": branch_holdup["superheat_dock_k"],
            "refrigerant_raw_superheat_k": ref["superheat_k"],
            "refrigerant_raw_cascade_superheat_k": ref["superheat_cascade_k"],
            "refrigerant_raw_dock_superheat_k": ref["superheat_dock_k"],
            "refrigerant_compressor_work_w": ref["w_ref_comp"],
            "refrigerant_compressor_isentropic_work_w": ref["w_ref_isentropic"],
            "refrigerant_compressor_eta_is_target": ref["eta_is_target"],
            "refrigerant_compressor_eta_is_effective": ref["eta_is_effective"],
            "cop_system": cop,
            "cop_room_only": cop_room_only,
            "base_room_load_w": self._base_room_load_w(time_s),
            "base_dock_load_w": self._base_dock_load_w(time_s),
            "infiltration_magnitude_w": infiltration_cfg.get("resolved_magnitude_w", infiltration_cfg.get("magnitude_w", 0.0)),
            "infiltration_room_w": infiltration["room_w"],
            "infiltration_dock_w": infiltration["dock_w"],
            "infiltration_q_m3_s": infiltration.get("q_m3_s", 0.0),
            "infiltration_q_unprotected_m3_s": infiltration.get("q_unprotected_m3_s", infiltration.get("q_m3_s", 0.0)),
            "infiltration_sensible_w": infiltration.get("q_sensible_w", 0.0),
            "infiltration_latent_w": infiltration.get("q_latent_w", 0.0),
            "infiltration_total_w": infiltration.get("q_total_w", infiltration["room_w"]),
            "infiltration_cumulative_volume_m3": infiltration.get("cumulative_volume_m3", 0.0),
            "infiltration_velocity_m_s": infiltration.get("velocity_m_s", 0.0),
            "infiltration_region_density_kg_m3": infiltration.get("region_density_kg_m3", 0.0),
            "infiltration_indoor_density_kg_m3": infiltration.get("rho_indoor_kg_m3", 0.0),
            "infiltration_outdoor_density_kg_m3": infiltration.get("rho_outdoor_kg_m3", 0.0),
            "infiltration_room_humidity_ratio_kg_kg_da": infiltration.get("omega_room_kg_kg_da", 0.0),
            "infiltration_outdoor_humidity_ratio_kg_kg_da": infiltration.get("omega_outdoor_kg_kg_da", 0.0),
            "infiltration_stage": infiltration.get("stage", 0.0),
            "infiltration_door_open_fraction": infiltration.get("door_open_fraction", 0.0),
            "infiltration_effective_length_m": infiltration.get("effective_length_m", 0.0),
            "infiltration_effective_volume_m3": infiltration.get("effective_volume_m3", 0.0),
            "infiltration_maximum_effective_length_m": infiltration.get("maximum_effective_length_m", 0.0),
            "load_w": self.load_w(time_s),
            "dock_load_w": self.dock_load_w(time_s),
        }
        state_vector = [room_c, sink_c, dock_c, self._room_moisture.humidity_ratio]
        if self.high_pressure_receiver_enabled():
            state_vector.append(receiver_mass_kg)
        if self.lpr_subcooling_control_enabled():
            state_vector.append(subcooling_k)
        return StepResult(values=values, state_vector=np.array(state_vector, dtype=float))
