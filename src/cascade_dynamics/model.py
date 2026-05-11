from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from CoolProp.CoolProp import PropsSI

from .compressor_map import ammonia_compressor_map, mass_flow_from_isentropic_head
from .components import compressor_actual_enthalpy, positive_lmtd, turbine_actual_enthalpy
from .fluids import h_ps, h_refrigerant_liquid, h_tp, p_sat, s_tp


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
        cfg = self.cfg.get("disturbances", {}).get("infiltration", {})
        if not cfg.get("enabled", False):
            return {"room_w": 0.0, "dock_w": 0.0}

        fraction = self._delayed_trapezoid_fraction(
            time_s,
            start_s=cfg["start_time_s"],
            delay_s=cfg.get("delay_s", 0.0),
            ramp_s=cfg.get("ramp_time_s", 30.0),
            hold_s=cfg.get("hold_time_s", 60.0),
        )
        magnitude_w = cfg.get("resolved_magnitude_w", cfg.get("magnitude_w", 0.0)) * fraction
        return {
            "room_w": magnitude_w * cfg.get("room_fraction", 1.0),
            "dock_w": magnitude_w * cfg.get("dock_fraction", -1.0),
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

        h1 = h_tp(t6_k, p1, self.air_fluid)
        s1 = s_tp(t6_k, p1, self.air_fluid)
        h2s = h_ps(p2, s1, self.air_fluid)
        compressor_head_is = h2s - h1
        rho1 = float(PropsSI("D", "T", t6_k, "P", p1, self.air_fluid))
        m_air = mass_flow_from_isentropic_head(air_cfg["compressor_mass_flow"], compressor_head_is, rho1)
        h2 = compressor_actual_enthalpy(h1, h2s, air_cfg["compressor_eta_is"])
        t2_k = float(PropsSI("T", "P", p2, "H", h2, self.air_fluid))
        w_air_comp = m_air * (h2 - h1)

        h4 = h_tp(t4_k, p2, self.air_fluid)
        s4 = s_tp(t4_k, p2, self.air_fluid)
        h5s = h_ps(p1, s4, self.air_fluid)
        h5 = turbine_actual_enthalpy(h4, h5s, air_cfg["turbine_eta_is"])
        t5_k = float(PropsSI("T", "P", p1, "H", h5, self.air_fluid))

        h3 = h_tp(t3_k, p2, self.air_fluid)
        h6 = h_tp(t6_k, p1, self.air_fluid)
        h_room = h_tp(room_k, p1, self.air_fluid)

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

        res = np.array(
            [
                room_c - prev_room_c - dt_s * (self.load_w(time_s) - air["q_room"]) / caps["room_capacitance_j_k"],
                dock_c - prev_dock_c - dt_s * (self.dock_load_w(time_s) - q_dock_evap) / caps["dock_capacitance_j_k"],
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
        ref = self._evaluate_refrigerant_cycle(tevap_k, tcond_k, m_ref, air["q_cascade"], q_dock)
        air_input_power = (air["w_air_comp"] - air["w_air_turb"]) / max(air_cfg.get("combined_drive_efficiency", 1.0), 1.0e-6)
        useful_cooling = air["q_room"] + q_dock
        cop = useful_cooling / max(air_input_power + ref["w_ref_comp"], 1.0)
        cop_room_only = air["q_room"] / max(air_input_power + ref["w_ref_comp"], 1.0)
        t2_c = air["t2_k"] - KELVIN_OFFSET
        t5_c = air["t5_k"] - KELVIN_OFFSET
        t7_c = ref["t7_k"] - KELVIN_OFFSET
        disturbance = self.infiltration_disturbance_w(time_s)
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
            "air_compressor_volumetric_flow_m3_s": air["m_air"] / max(air["rho1"], 1.0e-9),
            "air_compressor_isentropic_head_j_kg": air["compressor_head_is"],
            "air_compressor_isentropic_head_m": air["compressor_head_is"] / 9.80665,
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
            "infiltration_room_w": disturbance["room_w"],
            "infiltration_dock_w": disturbance["dock_w"],
            "load_w": self.load_w(time_s),
            "dock_load_w": self.dock_load_w(time_s),
        }
        return StepResult(values=values, state_vector=np.array([room_c, sink_c, dock_c], dtype=float))
