from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from CoolProp.CoolProp import PropsSI

from .compressor_map import ammonia_compressor_map, mass_flow_from_isentropic_head
from .components import compressor_actual_enthalpy, positive_lmtd, turbine_actual_enthalpy
from .fluids import h_refrigerant_liquid, p_sat
from .humid_air import (
    humid_air_state,
    saturated_room_humidity_ratio,
    state_at_enthalpy,
    state_at_entropy,
)
from .infiltration import TianInfiltrationState, advance_tian_infiltration, air_density, zero_infiltration_result


KELVIN_OFFSET = 273.15


@dataclass
class StepResult:
    values: dict[str, float]
    state_vector: np.ndarray


class CascadeSystemModel:
    def __init__(self, config: dict):
        self.cfg = config
        self.air_fluid = config["fluids"]["air"]
        self.ref_fluid = config["fluids"]["refrigerant"]
        self._infiltration_state = TianInfiltrationState()
        self._current_infiltration = zero_infiltration_result()

    def _infiltration_cfg(self) -> dict:
        return self.cfg.get("disturbances", {}).get("infiltration", {})

    def _uses_tian_infiltration(self) -> bool:
        cfg = self._infiltration_cfg()
        mode = str(cfg.get("model", cfg.get("magnitude_mode", ""))).lower()
        return cfg.get("enabled", False) and mode in {"tian", "tian_unsteady", "tian_analytical"}

    def _room_humidity_ratio(self, room_k: float, pressure_pa: float) -> float | None:
        humidity_cfg = self.cfg["air_cycle"].get("humid_air", {})
        x_room = humidity_cfg.get("humidity_ratio")
        if x_room is not None:
            return float(x_room)
        relative_humidity = humidity_cfg.get("room_relative_humidity")
        if relative_humidity is not None:
            return saturated_room_humidity_ratio(room_k, pressure_pa, float(relative_humidity))
        return None

    def reset_infiltration_disturbance(self, room_c: float, dock_c: float) -> None:
        cfg = self._infiltration_cfg()
        if not self._uses_tian_infiltration():
            self._current_infiltration = zero_infiltration_result()
            return
        p_i = float(cfg.get("P_i", cfg.get("indoor_pressure_pa", self.cfg["air_cycle"]["p_low_pa"])))
        room_k = room_c + KELVIN_OFFSET
        omega_room = self._room_humidity_ratio(room_k, p_i)
        rho_i = air_density(room_k, p_i, omega_room)
        self._infiltration_state = TianInfiltrationState(density_kg_m3=rho_i)
        self._current_infiltration = zero_infiltration_result()
        self._current_infiltration["region_density_kg_m3"] = self._infiltration_state.density_kg_m3

    def advance_infiltration_disturbance(self, time_s: float, dt_s: float, previous_values: dict[str, float]) -> None:
        if not self._uses_tian_infiltration():
            return

        cfg = self._infiltration_cfg()
        room_c = float(previous_values["room_c"])
        outdoor_source = str(cfg.get("outdoor_source", "dock")).lower()
        if outdoor_source == "ambient":
            outdoor_c = float(self.cfg["boundary_conditions"]["ambient_c"])
        elif outdoor_source == "fixed":
            outdoor_c = float(cfg.get("outdoor_c", cfg.get("T_o_c", self.cfg["boundary_conditions"]["ambient_c"])))
        else:
            outdoor_c = float(previous_values.get("dock_c", self.cfg["boundary_conditions"]["dock_initial_c"]))

        p_i = float(cfg.get("P_i", cfg.get("indoor_pressure_pa", self.cfg["air_cycle"]["p_low_pa"])))
        p_o = float(cfg.get("P_o", cfg.get("outdoor_pressure_pa", p_i)))
        room_k = room_c + KELVIN_OFFSET
        omega_room = self._room_humidity_ratio(room_k, p_i)

        self._current_infiltration = advance_tian_infiltration(
            cfg,
            self._infiltration_state,
            time_s=time_s,
            dt_s=dt_s,
            room_c=room_c,
            outdoor_c=outdoor_c,
            pressure_i_pa=p_i,
            pressure_o_pa=p_o,
            omega_room=omega_room,
        )

    def _constraint_penalty(self, unknowns: np.ndarray) -> np.ndarray:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref, dock_c = unknowns
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
            max(0.0, 1.0e-5 - m_ref),
            max(0.0, m_ref - 5.0),
        ]
        penalty_sum = sum(penalties)
        if penalty_sum < 1.0e-7:
            return np.zeros(9, dtype=float)
        scale = 1.0e6
        p0 = 1.0e3 + scale * penalty_sum
        return np.full(9, p0, dtype=float)

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
        exchange_w = magnitude_w
        return {
            "room_w": exchange_w,
            "dock_w": -exchange_w,
            "q_m3_s": 0.0,
            "q_sensible_w": exchange_w,
            "q_latent_w": 0.0,
            "q_total_w": exchange_w,
            "cumulative_volume_m3": 0.0,
            "velocity_m_s": 0.0,
            "region_density_kg_m3": 0.0,
            "stage": 0.0,
            "door_open_fraction": fraction,
            "effective_length_m": 0.0,
        }

    def load_w(self, time_s: float) -> float:
        return self._base_room_load_w(time_s) + self.infiltration_disturbance_w(time_s)["room_w"]

    def dock_load_w(self, time_s: float) -> float:
        return self._base_dock_load_w(time_s) + self.infiltration_disturbance_w(time_s)["dock_w"]

    def startup_evaluation(self, unknowns: np.ndarray, time_s: float) -> tuple[np.ndarray, dict[str, float]]:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref, dock_c = unknowns
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
        q_dock_evap = self._evaluate_dock_evaporator(dock_c, tevap_c)
        infiltration = self.infiltration_disturbance_w(time_s)
        ref = self._evaluate_refrigerant_cycle(tevap_k, tcond_k, m_ref, air["q_cascade"], q_dock_evap)

        t2_c = air["t2_k"] - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - tevap_c, t3_c - tevap_c)
        condenser_lmtd = positive_lmtd(tcond_c - ambient_c, tcond_c - sink_c)

        q_reg_ua = air_cfg["regenerator_ua_w_k"] * reg_lmtd
        q_cascade_ua = vcc_cfg["cascade_ua_w_k"] * cascade_lmtd
        q_cond_ua = vcc_cfg["condenser_ua_w_k"] * condenser_lmtd
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - ambient_c)

        balances = np.array(
            [
                air["q_room"] - self.load_w(time_s),
                q_dock_evap - self.dock_load_w(time_s),
                ref["q_cond"] - sink_rejection,
                air["q_reg_hot"] - air["q_reg_cold"],
                air["q_reg_hot"] - q_reg_ua,
                air["q_cascade"] - q_cascade_ua,
                ref["q_cond"] - q_cond_ua,
                m_ref - ref["m_ref_valve"],
                m_ref - ref["map_m_ref_kg_s"],
            ],
            dtype=float,
        )

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
            "m_air_kg_s": air["m_air"],
            "infiltration_room_w": infiltration["room_w"],
            "infiltration_dock_w": infiltration["dock_w"],
            "humidity_ratio_room_kg_kg_da": air["humidity_ratio_room"],
            "humidity_ratio_supply_vapor_kg_kg_da": air["humidity_ratio_5_vapor"],
            "humidity_ratio_supply_ice_kg_kg_da": air["humidity_ratio_5_ice"],
            "ice_mass_flow_kg_s": air["ice_mass_flow"],
            "q_dock_w": q_dock_evap,
            "refrigerant_superheat_k": ref["superheat_k"],
            "air_compressor_isentropic_head_j_kg": air["compressor_head_is"],
            "air_compressor_isentropic_head_m": air["compressor_head_is"] / 9.80665,
        }
        return balances, metrics

    def _evaluate_air_cycle(self, room_k: float, t3_k: float, t4_k: float, t6_k: float) -> dict[str, float]:
        air_cfg = self.cfg["air_cycle"]
        p1 = air_cfg["p_low_pa"]
        p2 = p1 * air_cfg["pressure_ratio"]
        humidity_cfg = air_cfg.get("humid_air", {})
        relative_humidity = humidity_cfg.get("room_relative_humidity", 1.0)
        x_room = humidity_cfg.get("humidity_ratio")
        if x_room is None:
            x_room = saturated_room_humidity_ratio(room_k, p1, relative_humidity)
        x_room = float(x_room)

        state_room = humid_air_state(room_k, p1, x_room)
        state1 = humid_air_state(t6_k, p1, x_room)
        state2s = state_at_entropy(p2, x_room, state1.entropy_j_kg_da_k)
        h1 = state1.enthalpy_j_kg_da
        h2s = state2s.enthalpy_j_kg_da
        compressor_head_is = h2s - h1
        rho1 = state1.dry_air_density_kg_m3
        m_air = mass_flow_from_isentropic_head(air_cfg["compressor_mass_flow"], compressor_head_is, rho1)
        h2 = compressor_actual_enthalpy(h1, h2s, air_cfg["compressor_eta_is"])
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
            "rho1": rho1,
            "rho1_mixture": state1.mixture_density_kg_m3,
            "humidity_ratio_room": x_room,
            "humidity_ratio_5_vapor": state5.vapor_humidity_ratio,
            "humidity_ratio_5_ice": state5.ice_humidity_ratio,
            "ice_mass_flow": m_air * state5.ice_humidity_ratio,
            "compressor_head_is": compressor_head_is,
            "q_cascade": m_air * (h2 - h3),
            "q_reg_hot": m_air * (h3 - h4),
            "q_reg_cold": m_air * (h6 - h_room),
            "q_room": m_air * (h_room - h5),
            "w_air_comp": w_air_comp,
            "w_air_turb": m_air * (h4 - h5),
        }

    def _evaluate_refrigerant_cycle(self, tevap_k: float, tcond_k: float, m_ref: float, q_cascade: float, q_dock: float) -> dict[str, float]:
        vcc_cfg = self.cfg["vcc_cycle"]
        p_evap = p_sat(tevap_k, self.ref_fluid)
        p_cond = p_sat(tcond_k, self.ref_fluid)
        t9_k = tcond_k - vcc_cfg["subcooling_k"]
        h9 = h_refrigerant_liquid(t9_k, p_cond, self.ref_fluid, vcc_cfg["subcooling_k"])
        h10 = h9
        evap_total = q_cascade + q_dock
        h7 = h10 + evap_total / max(m_ref, 1.0e-6)
        t7_k = float(PropsSI("T", "P", p_evap, "H", h7, self.ref_fluid))
        superheat_k = t7_k - tevap_k
        compressor_cfg = vcc_cfg["compressor"]
        map_result = ammonia_compressor_map(
            tcond_k,
            tevap_k,
            compressor_cfg["speed_rpm"],
            check_range=compressor_cfg.get("check_range", True),
        )
        w_ref_comp = map_result["P_W"]
        q_cond = evap_total + w_ref_comp
        valve_cfg = vcc_cfg["expansion_valve"]
        m_ref_valve = valve_cfg["flow_coefficient_kg_s_pa"] * valve_cfg["opening"] * max(p_cond - p_evap, 0.0)
        return {
            "p_evap": p_evap,
            "p_cond": p_cond,
            "h9": h9,
            "h10": h10,
            "h7": h7,
            "t7_k": t7_k,
            "superheat_k": superheat_k,
            "q_evap_total": evap_total,
            "q_cond": q_cond,
            "w_ref_comp": w_ref_comp,
            "map_q_w": map_result["Q_W"],
            "map_m_ref_kg_s": map_result["mdot_kg_s"],
            "map_cop": map_result["COP"],
            "m_ref_valve": m_ref_valve,
            "speed_rpm": float(compressor_cfg["speed_rpm"]),
        }

    def _evaluate_dock_evaporator(self, dock_c: float, tevap_c: float) -> float:
        ua_w_k = float(self.cfg["vcc_cycle"].get("dock_evaporator_ua_w_k", 0.0))
        return ua_w_k * max(dock_c - tevap_c, 0.0)

    def residual(self, unknowns: np.ndarray, prev_state: np.ndarray, time_s: float, dt_s: float) -> np.ndarray:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref, dock_c = unknowns
        prev_room_c, prev_sink_c, prev_dock_c = prev_state
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
            q_dock_evap = self._evaluate_dock_evaporator(dock_c, tevap_c)
            ref = self._evaluate_refrigerant_cycle(tevap_k, tcond_k, m_ref, air["q_cascade"], q_dock_evap)
        except ValueError:
            return np.full(9, 1.0e9, dtype=float)

        t2_c = air["t2_k"] - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - tevap_c, t3_c - tevap_c)
        condenser_lmtd = positive_lmtd(tcond_c - ambient_c, tcond_c - sink_c)

        q_reg_ua = air_cfg["regenerator_ua_w_k"] * reg_lmtd
        q_cascade_ua = vcc_cfg["cascade_ua_w_k"] * cascade_lmtd
        q_cond_ua = vcc_cfg["condenser_ua_w_k"] * condenser_lmtd
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - ambient_c)
        room_load_w = self.load_w(time_s)
        dock_load_w = self.dock_load_w(time_s)

        res = np.array(
            [
                room_c - prev_room_c - dt_s * (room_load_w - air["q_room"]) / caps["room_capacitance_j_k"],
                dock_c - prev_dock_c - dt_s * (dock_load_w - q_dock_evap) / caps["dock_capacitance_j_k"],
                sink_c - prev_sink_c - dt_s * (ref["q_cond"] - sink_rejection) / caps["sink_capacitance_j_k"],
                air["q_reg_hot"] - air["q_reg_cold"],
                air["q_reg_hot"] - q_reg_ua,
                air["q_cascade"] - q_cascade_ua,
                ref["q_cond"] - q_cond_ua,
                m_ref - ref["m_ref_valve"],
                m_ref - ref["map_m_ref_kg_s"],
            ],
            dtype=float,
        )
        return res

    def steady_state_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        steady_state = np.array([unknowns[0], unknowns[1], unknowns[8]], dtype=float)
        return self.residual(unknowns, steady_state, time_s, dt_s=1.0)

    def startup_balance_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        balances, _ = self.startup_evaluation(unknowns, time_s)
        return balances

    def startup_metrics(self, unknowns: np.ndarray, time_s: float) -> dict[str, float]:
        _, metrics = self.startup_evaluation(unknowns, time_s)
        return metrics

    def post_process(self, unknowns: np.ndarray, time_s: float) -> StepResult:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, tcond_c, m_ref, dock_c = unknowns
        room_k = room_c + KELVIN_OFFSET
        sink_k = sink_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET
        tcond_k = tcond_c + KELVIN_OFFSET
        air_cfg = self.cfg["air_cycle"]
        air = self._evaluate_air_cycle(room_k, t3_k, t4_k, t6_k)
        q_dock = self._evaluate_dock_evaporator(dock_c, tevap_c)
        infiltration = self.infiltration_disturbance_w(time_s)
        ref = self._evaluate_refrigerant_cycle(tevap_k, tcond_k, m_ref, air["q_cascade"], q_dock)
        air_input_power = (air["w_air_comp"] - air["w_air_turb"]) / max(air_cfg.get("combined_drive_efficiency", 1.0), 1.0e-6)
        useful_cooling = air["q_room"] + q_dock
        cop = useful_cooling / max(air_input_power + ref["w_ref_comp"], 1.0)
        cop_room_only = air["q_room"] / max(air_input_power + ref["w_ref_comp"], 1.0)
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
            "q_useful_w": useful_cooling,
            "q_cascade_w": air["q_cascade"],
            "q_cond_w": ref["q_cond"],
            "w_air_comp_w": air["w_air_comp"],
            "w_air_turb_w": air["w_air_turb"],
            "w_air_input_w": air_input_power,
            "w_ref_comp_w": ref["w_ref_comp"],
            "m_ref_kg_s": m_ref,
            "m_air_kg_s": air["m_air"],
            "air_compressor_suction_density_kg_m3": air["rho1"],
            "air_compressor_suction_mixture_density_kg_m3": air["rho1_mixture"],
            "air_compressor_volumetric_flow_m3_s": air["m_air"] / max(air["rho1"], 1.0e-9),
            "air_compressor_isentropic_head_j_kg": air["compressor_head_is"],
            "air_compressor_isentropic_head_m": air["compressor_head_is"] / 9.80665,
            "humidity_ratio_room_kg_kg_da": air["humidity_ratio_room"],
            "humidity_ratio_supply_vapor_kg_kg_da": air["humidity_ratio_5_vapor"],
            "humidity_ratio_supply_ice_kg_kg_da": air["humidity_ratio_5_ice"],
            "ice_mass_flow_kg_s": air["ice_mass_flow"],
            "valve_flow_kg_s": ref["m_ref_valve"],
            "valve_opening": self.cfg["vcc_cycle"]["expansion_valve"]["opening"],
            "refrigerant_superheat_k": ref["superheat_k"],
            "refrigerant_compressor_speed_rpm": ref["speed_rpm"],
            "refrigerant_map_mass_flow_kg_s": ref["map_m_ref_kg_s"],
            "refrigerant_map_capacity_w": ref["map_q_w"],
            "refrigerant_map_cop": ref["map_cop"],
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
            "load_w": self.load_w(time_s),
            "dock_load_w": self.dock_load_w(time_s),
        }
        return StepResult(values=values, state_vector=np.array([room_c, sink_c, dock_c], dtype=float))
