from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def update(self, measurements: dict[str, float], plant_config: dict[str, Any], dt_s: float) -> float:
        if not self.enabled:
            return float(get_path(plant_config, self.cfg["actuator_path"]))

        measured_value = measurements[self.cfg["measurement"]]
        setpoint = float(self.cfg["setpoint"])
        error = measured_value - setpoint
        if self.cfg.get("action", "direct") == "reverse":
            error = -error

        proposed_integral = self.state.integral + error * dt_s

        bias = float(self.cfg.get("bias", get_path(plant_config, self.cfg["actuator_path"])))
        gain = float(self.cfg["gain"])
        ti_min = float(self.cfg["Ti_min"])

        raw_output = bias + gain * error
        raw_output += gain * (proposed_integral / (ti_min * 60.0))

        output = min(max(raw_output, float(self.cfg["u_min"])), float(self.cfg["u_max"]))
        if output == raw_output or not self.cfg.get("anti_windup", True):
            self.state.integral = proposed_integral
        set_path(plant_config, self.cfg["actuator_path"], output)
        return output


class ControlSystem:
    def __init__(self, config: dict[str, Any]):
        control_cfg = config.get("control", {})
        self.enabled = control_cfg.get("enabled", False)
        self.controllers = [PIDController(item) for item in control_cfg.get("controllers", [])]

    def update(self, measurements: dict[str, float], plant_config: dict[str, Any], dt_s: float) -> dict[str, float]:
        if not self.enabled:
            return {}
        outputs: dict[str, float] = {}
        for controller in self.controllers:
            outputs[f"pid_{controller.name}_output"] = controller.update(measurements, plant_config, dt_s)
        return outputs
