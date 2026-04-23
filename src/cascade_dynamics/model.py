from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from CoolProp.CoolProp import PropsSI

from .compressor_map import mass_flow_from_isentropic_head
from .components import compressor_actual_enthalpy, positive_lmtd, turbine_actual_enthalpy
from .fluids import h_ps, h_refrigerant_liquid, h_tp, p_sat, s_tp, t_sat


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
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, m_ref = unknowns
        penalties = [
            max(0.0, -83.15 - room_c),
            max(0.0, room_c - 46.85),
            max(0.0, -3.15 - sink_c),
            max(0.0, sink_c - 86.85),
            max(0.0, -73.15 - tevap_c),
            max(0.0, tevap_c - 46.85),
            max(0.0, 1.0e-5 - m_ref),
            max(0.0, m_ref - 5.0),
        ]
        penalty_sum = sum(penalties)
        if penalty_sum < 1.0e-7:
            return np.zeros(7, dtype=float)
        scale = 1.0e6
        p0 = 1.0e3 + scale * penalty_sum
        return np.full(7, p0, dtype=float)

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
        magnitude_w = cfg["magnitude_w"] * fraction
        return {
            "room_w": magnitude_w * cfg.get("room_fraction", 1.0),
            "dock_w": magnitude_w * cfg.get("dock_fraction", -1.0),
        }

    def load_w(self, time_s: float) -> float:
        return self._base_room_load_w(time_s) + self.infiltration_disturbance_w(time_s)["room_w"]

    def dock_load_w(self, time_s: float) -> float:
        return self._base_dock_load_w(time_s) + self.infiltration_disturbance_w(time_s)["dock_w"]

    def residual(self, unknowns: np.ndarray, prev_state: np.ndarray, time_s: float, dt_s: float) -> np.ndarray:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, m_ref = unknowns
        prev_room_c, prev_sink_c = prev_state
        room_k = room_c + KELVIN_OFFSET
        sink_k = sink_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET

        penalty = self._constraint_penalty(unknowns)
        if np.any(penalty > 0.0):
            return penalty

        air_cfg = self.cfg["air_cycle"]
        vcc_cfg = self.cfg["vcc_cycle"]
        bc = self.cfg["boundary_conditions"]
        caps = self.cfg["thermal_masses"]

        p1 = air_cfg["p_low_pa"]
        p2 = p1 * air_cfg["pressure_ratio"]
        ambient_c = bc["ambient_c"]
        ambient_k = ambient_c + KELVIN_OFFSET

        try:
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

            q_cascade = m_air * (h2 - h3)
            q_reg_hot = m_air * (h3 - h4)
            q_reg_cold = m_air * (h6 - h_room)
            q_room = m_air * (h_room - h5)
            w_air_turb = m_air * (h4 - h5)

            p_evap = p_sat(tevap_k, self.ref_fluid)
            p_cond = self._refrigerant_discharge_pressure(vcc_cfg, p_evap)
            q_dock = self.dock_load_w(time_s)
            evap_total = q_cascade + q_dock
            tcond_k = t_sat(p_cond, self.ref_fluid)
            t9_k = tcond_k - vcc_cfg["subcooling_k"]
            h9 = h_refrigerant_liquid(t9_k, p_cond, self.ref_fluid, vcc_cfg["subcooling_k"])
            h10 = h9
            h7 = h10 + evap_total / max(m_ref, 1.0e-6)
            s7 = float(PropsSI("S", "P", p_evap, "H", h7, self.ref_fluid))
            h8s = h_ps(p_cond, s7, self.ref_fluid)
            h8 = compressor_actual_enthalpy(h7, h8s, vcc_cfg["compressor_eta_is"])
            q_cond = m_ref * (h8 - h9)
        except ValueError:
            return np.full(7, 1.0e9, dtype=float)

        t2_c = t2_k - KELVIN_OFFSET
        tcond_c = tcond_k - KELVIN_OFFSET
        reg_lmtd = positive_lmtd(t3_c - t6_c, t4_c - room_c)
        cascade_lmtd = positive_lmtd(t2_c - tevap_c, t3_c - tevap_c)
        condenser_lmtd = positive_lmtd(tcond_c - ambient_c, tcond_c - sink_c)

        q_reg_ua = air_cfg["regenerator_ua_w_k"] * reg_lmtd
        q_cascade_ua = vcc_cfg["cascade_ua_w_k"] * cascade_lmtd
        q_cond_ua = vcc_cfg["condenser_ua_w_k"] * condenser_lmtd
        sink_rejection = bc["sink_m_dot_kg_s"] * bc["sink_cp_j_kg_k"] * (sink_c - ambient_c)
        valve_cfg = vcc_cfg["expansion_valve"]
        m_ref_valve = valve_cfg["flow_coefficient_kg_s_pa"] * valve_cfg["opening"] * max(p_cond - p_evap, 0.0)

        res = np.array(
            [
                q_room - self.load_w(time_s),
                sink_c - prev_sink_c - dt_s * (q_cond - sink_rejection) / caps["sink_capacitance_j_k"],
                q_reg_hot - q_reg_cold,
                q_reg_hot - q_reg_ua,
                q_cascade - q_cascade_ua,
                q_cond - q_cond_ua,
                m_ref - m_ref_valve,
            ],
            dtype=float,
        )
        return res

    def steady_state_residual(self, unknowns: np.ndarray, time_s: float) -> np.ndarray:
        steady_state = unknowns[:2]
        return self.residual(unknowns, steady_state, time_s, dt_s=1.0)

    def post_process(self, unknowns: np.ndarray, time_s: float) -> StepResult:
        room_c, sink_c, t3_c, t4_c, t6_c, tevap_c, m_ref = unknowns
        room_k = room_c + KELVIN_OFFSET
        sink_k = sink_c + KELVIN_OFFSET
        t3_k = t3_c + KELVIN_OFFSET
        t4_k = t4_c + KELVIN_OFFSET
        t6_k = t6_c + KELVIN_OFFSET
        tevap_k = tevap_c + KELVIN_OFFSET
        air_cfg = self.cfg["air_cycle"]
        vcc_cfg = self.cfg["vcc_cycle"]
        bc = self.cfg["boundary_conditions"]
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
        q_room = m_air * (h_room - h5)
        q_cascade = m_air * (h2 - h3)
        w_air_turb = m_air * (h4 - h5)

        p_evap = p_sat(tevap_k, self.ref_fluid)
        p_cond = self._refrigerant_discharge_pressure(vcc_cfg, p_evap)
        tcond_k = t_sat(p_cond, self.ref_fluid)
        t9_k = tcond_k - vcc_cfg["subcooling_k"]
        h9 = h_refrigerant_liquid(t9_k, p_cond, self.ref_fluid, vcc_cfg["subcooling_k"])
        h10 = h9
        h7 = h10 + (q_cascade + self.dock_load_w(time_s)) / max(m_ref, 1.0e-6)
        t7_k = float(PropsSI("T", "P", p_evap, "H", h7, self.ref_fluid))
        s7 = float(PropsSI("S", "P", p_evap, "H", h7, self.ref_fluid))
        h8s = h_ps(p_cond, s7, self.ref_fluid)
        h8 = compressor_actual_enthalpy(h7, h8s, vcc_cfg["compressor_eta_is"])
        valve_cfg = vcc_cfg["expansion_valve"]
        q_dock = self.dock_load_w(time_s)
        q_cond = m_ref * (h8 - h9)
        w_ref_comp = m_ref * (h8 - h7)
        valve_flow = m_ref
        air_input_power = (w_air_comp - w_air_turb) / max(air_cfg.get("combined_drive_efficiency", 1.0), 1.0e-6)
        useful_cooling = q_room + q_dock
        cop = useful_cooling / max(air_input_power + w_ref_comp, 1.0)
        cop_room_only = q_room / max(air_input_power + w_ref_comp, 1.0)
        t2_c = t2_k - KELVIN_OFFSET
        t5_c = t5_k - KELVIN_OFFSET
        t7_c = t7_k - KELVIN_OFFSET
        tcond_c = tcond_k - KELVIN_OFFSET

        values = {
            "time_s": time_s,
            "room_c": room_c,
            "dock_c": bc["dock_initial_c"],
            "sink_c": sink_c,
            "t2_c": t2_c,
            "t3_c": t3_c,
            "t4_c": t4_c,
            "t5_c": t5_c,
            "t6_c": t6_c,
            "t7_c": t7_c,
            "tevap_c": tevap_c,
            "tcond_c": tcond_c,
            "q_room_w": q_room,
            "q_dock_w": q_dock,
            "q_useful_w": useful_cooling,
            "q_cascade_w": q_cascade,
            "q_cond_w": q_cond,
            "w_air_comp_w": w_air_comp,
            "w_air_turb_w": w_air_turb,
            "w_air_input_w": air_input_power,
            "w_ref_comp_w": w_ref_comp,
            "m_ref_kg_s": m_ref,
            "m_air_kg_s": m_air,
            "air_compressor_suction_density_kg_m3": rho1,
            "air_compressor_volumetric_flow_m3_s": m_air / max(rho1, 1.0e-9),
            "air_compressor_isentropic_head_j_kg": compressor_head_is,
            "air_compressor_isentropic_head_m": compressor_head_is / 9.80665,
            "valve_flow_kg_s": valve_flow,
            "valve_opening": valve_cfg["opening"],
            "cop_system": cop,
            "cop_room_only": cop_room_only,
            "base_room_load_w": self._base_room_load_w(time_s),
            "base_dock_load_w": self._base_dock_load_w(time_s),
            "infiltration_room_w": self.infiltration_disturbance_w(time_s)["room_w"],
            "infiltration_dock_w": self.infiltration_disturbance_w(time_s)["dock_w"],
            "load_w": self.load_w(time_s),
            "dock_load_w": self.dock_load_w(time_s),
        }
        return StepResult(values=values, state_vector=np.array([room_c, sink_c], dtype=float))

    def _refrigerant_discharge_pressure(self, vcc_cfg: dict, p_evap: float) -> float:
        if "compressor_pressure_ratio" in vcc_cfg:
            return p_evap * vcc_cfg["compressor_pressure_ratio"]
        return p_evap
