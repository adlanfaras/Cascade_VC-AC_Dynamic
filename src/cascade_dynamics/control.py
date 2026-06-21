from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .infiltration import door_open_fraction, scheduled_precool_fraction


def get_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[part]
    return current


def set_path(data: dict[str, Any], path: str, value: float) -> None:
    current: Any = data
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = float(value)


def _as_string_set(value: Any, default: tuple[str, ...] = ()) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


@dataclass
class PIDState:
    integral: float = 0.0


class PIDController:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.state = PIDState()

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def enabled(self) -> bool:
        return self.cfg.get("enabled", True)

    def _apply_actuator_dynamics(self, target_output: float, plant_config: dict[str, Any], dt_s: float) -> float:
        previous_output = float(get_path(plant_config, self.cfg["actuator_path"]))
        output = target_output

        time_constant_s = float(self.cfg.get("actuator_time_constant_s", 0.0))
        if time_constant_s > 0.0:
            alpha = min(max(dt_s / time_constant_s, 0.0), 1.0)
            output = previous_output + alpha * (target_output - previous_output)

        rate_limit_per_s = self.cfg.get("actuator_rate_limit_per_s")
        if rate_limit_per_s is not None:
            max_delta = abs(float(rate_limit_per_s)) * dt_s
            delta = min(max(output - previous_output, -max_delta), max_delta)
            output = previous_output + delta

        return min(max(output, float(self.cfg["u_min"])), float(self.cfg["u_max"]))

    def update(
        self,
        measurements: dict[str, float],
        plant_config: dict[str, Any],
        dt_s: float,
        setpoint_offset: float = 0.0,
    ) -> float:
        if not self.enabled:
            return float(get_path(plant_config, self.cfg["actuator_path"]))

        measured_value = measurements[self.cfg["measurement"]]
        setpoint = float(self.cfg["setpoint"]) + float(setpoint_offset)
        error = measured_value - setpoint
        if self.cfg.get("action", "direct") == "reverse":
            error = -error

        proposed_integral = self.state.integral + error * dt_s

        bias = float(self.cfg.get("bias", get_path(plant_config, self.cfg["actuator_path"])))
        gain = float(self.cfg["gain"])
        ti_s = max(float(self.cfg["Ti_min"]) * 60.0, 1.0e-9)

        raw_output = bias + gain * error
        raw_output += gain * (proposed_integral / ti_s)

        target_output = min(max(raw_output, float(self.cfg["u_min"])), float(self.cfg["u_max"]))
        output = self._apply_actuator_dynamics(target_output, plant_config, dt_s)
        if not self.cfg.get("anti_windup", True):
            self.state.integral = proposed_integral
        elif abs(gain) > 1.0e-12:
            self.state.integral = (output - bias - gain * error) * ti_s / gain
        elif abs(output - raw_output) <= 1.0e-12:
            self.state.integral = proposed_integral
        set_path(plant_config, self.cfg["actuator_path"], output)
        return output


class ProactiveController:
    def __init__(self, config: dict[str, Any]):
        control_cfg = config.get("control", {})
        cfg = control_cfg.get("proactive", {})
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.enabled = bool(self.cfg.get("enabled", False))
        disturbances = config.get("disturbances", {})
        infiltration_cfg = disturbances.get("infiltration", {})
        self.infiltration_cfg = infiltration_cfg if isinstance(infiltration_cfg, dict) else {}

        self.precool_enabled = bool(self.cfg.get("precool_enabled", self.cfg.get("enabled", False)))
        self.precool_lead_time_s = max(float(self.cfg.get("precool_lead_time_s", 0.0)), 0.0)
        self.precool_depth_k = max(float(self.cfg.get("precool_depth_k", 0.0)), 0.0)
        self.precool_measurements = _as_string_set(
            self.cfg.get(
                "precool_measurements",
                ("room_c", "refrigerating_temperature_c", "room_k", "refrigerating_temperature_k"),
            )
        )
        self.precool_controller_names = _as_string_set(self.cfg.get("precool_controller_names", ()))

        self.boost_enabled = bool(self.cfg.get("feedforward_enabled", self.cfg.get("enabled", False)))
        self.default_speed_boost_fraction = float(
            self.cfg.get("compressor_boost_fraction", self.cfg.get("speed_boost_fraction", 0.0))
        )
        self.boost_controller_names = _as_string_set(self.cfg.get("boost_controller_names", ()))
        self.boost_actuator_paths = _as_string_set(self.cfg.get("boost_actuator_paths", ()))
        self.boost_by_controller: dict[str, float] = {}
        self.boost_by_actuator: dict[str, float] = {}
        for boost_cfg in self.cfg.get("feedforward_boosts", ()):
            if not isinstance(boost_cfg, dict):
                continue
            fraction = float(boost_cfg.get("fraction", boost_cfg.get("boost_fraction", 0.0)))
            if "controller_name" in boost_cfg:
                self.boost_by_controller[str(boost_cfg["controller_name"])] = fraction
            if "actuator_path" in boost_cfg:
                self.boost_by_actuator[str(boost_cfg["actuator_path"])] = fraction

    def current_time_s(self, measurements: dict[str, float], dt_s: float) -> float:
        return float(measurements.get("time_s", 0.0)) + float(dt_s)

    def precool_fraction(self, time_s: float) -> float:
        if not (self.enabled and self.precool_enabled and self.precool_depth_k > 0.0 and self.precool_lead_time_s > 0.0):
            return 0.0
        return scheduled_precool_fraction(self.infiltration_cfg, time_s, self.precool_lead_time_s)

    def door_fraction(self, time_s: float, measurements: dict[str, float]) -> float:
        if not self.enabled:
            return 0.0
        if self.infiltration_cfg:
            return door_open_fraction(self.infiltration_cfg, time_s)
        return float(measurements.get("infiltration_door_open_fraction", 0.0))

    def setpoint_offset(self, controller: PIDController, precool_fraction_value: float) -> float:
        if precool_fraction_value <= 0.0:
            return 0.0
        if self.precool_controller_names and controller.name not in self.precool_controller_names:
            return 0.0
        if not self.precool_controller_names and controller.cfg.get("measurement") not in self.precool_measurements:
            return 0.0
        return -self.precool_depth_k * precool_fraction_value

    def boost_fraction(self, controller: PIDController) -> float:
        if not (self.enabled and self.boost_enabled):
            return 0.0

        actuator_path = str(controller.cfg["actuator_path"])
        if controller.name in self.boost_by_controller:
            return float(self.boost_by_controller[controller.name])
        if actuator_path in self.boost_by_actuator:
            return float(self.boost_by_actuator[actuator_path])
        if self.boost_controller_names and controller.name in self.boost_controller_names:
            return self.default_speed_boost_fraction
        if self.boost_actuator_paths and actuator_path in self.boost_actuator_paths:
            return self.default_speed_boost_fraction
        if not self.boost_controller_names and not self.boost_actuator_paths and actuator_path.endswith("speed_rpm"):
            return self.default_speed_boost_fraction
        return 0.0

    def apply_feedforward_boost(
        self,
        controller: PIDController,
        feedback_output: float,
        plant_config: dict[str, Any],
        door_fraction_value: float,
    ) -> float:
        fraction = max(float(self.boost_fraction(controller)), 0.0)
        if fraction <= 0.0 or door_fraction_value <= 0.0:
            return feedback_output

        u_max = float(controller.cfg["u_max"])
        boosted_output = feedback_output + min(fraction, 1.0) * door_fraction_value * (u_max - feedback_output)
        boosted_output = min(max(boosted_output, float(controller.cfg["u_min"])), u_max)
        set_path(plant_config, controller.cfg["actuator_path"], boosted_output)
        return boosted_output


class ControlSystem:
    def __init__(self, config: dict[str, Any], frozen_actuator_paths: set[str] | None = None):
        control_cfg = config.get("control", {})
        self.enabled = control_cfg.get("enabled", False)
        self.frozen_actuator_paths = frozen_actuator_paths or set()
        self.controllers = [PIDController(item) for item in control_cfg.get("controllers", [])]
        self.proactive = ProactiveController(config)

    def update(self, measurements: dict[str, float], plant_config: dict[str, Any], dt_s: float) -> dict[str, float]:
        if not self.enabled:
            return {}
        outputs: dict[str, float] = {}
        time_s = self.proactive.current_time_s(measurements, dt_s)
        precool_fraction_value = self.proactive.precool_fraction(time_s)
        door_fraction_value = self.proactive.door_fraction(time_s, measurements)
        if self.proactive.enabled:
            outputs["proactive_precool_active"] = precool_fraction_value
            outputs["proactive_precool_offset_k"] = -self.proactive.precool_depth_k * precool_fraction_value
            outputs["proactive_door_open_fraction"] = door_fraction_value
        for controller in self.controllers:
            if controller.cfg["actuator_path"] in self.frozen_actuator_paths:
                outputs[f"pid_{controller.name}_output"] = float(get_path(plant_config, controller.cfg["actuator_path"]))
                outputs[f"pid_{controller.name}_frozen"] = 1.0
                continue
            setpoint_offset = self.proactive.setpoint_offset(controller, precool_fraction_value)
            feedback_output = controller.update(measurements, plant_config, dt_s, setpoint_offset=setpoint_offset)
            output = self.proactive.apply_feedforward_boost(
                controller,
                feedback_output,
                plant_config,
                door_fraction_value,
            )
            if setpoint_offset != 0.0:
                outputs[f"proactive_{controller.name}_setpoint_offset_k"] = setpoint_offset
                outputs[f"proactive_{controller.name}_setpoint"] = float(controller.cfg["setpoint"]) + setpoint_offset
            if output != feedback_output:
                outputs[f"pid_{controller.name}_feedback_output"] = feedback_output
                outputs[f"proactive_{controller.name}_feedforward_boost"] = output - feedback_output
            outputs[f"pid_{controller.name}_output"] = output
        return outputs
