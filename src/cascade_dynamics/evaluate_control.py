from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .fluids import DEFAULT_REFPROP_PATH
from .infiltration import scheduled_door_event_starts_between
from .simulation import run_simulation, save_csv


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _with_case_output(config: dict[str, Any], output_dir: Path, run_label: str, case_name: str) -> dict[str, Any]:
    case_config = deepcopy(config)
    output_cfg = case_config.setdefault("output", {})
    stem = f"{_safe_label(run_label)}_{_safe_label(case_name)}"
    output_cfg["csv_file"] = str(output_dir / f"{stem}.csv")
    output_cfg["plot_file"] = str(output_dir / f"{stem}.png")
    return case_config


def _room_setpoint_c(config: dict[str, Any]) -> float:
    for controller in config.get("control", {}).get("controllers", []):
        if controller.get("measurement") == "room_c" and "setpoint" in controller:
            return float(controller["setpoint"])
    targets = config.get("simulation", {}).get("startup_initialization", {}).get("targets", {})
    if "room_c" in targets:
        return float(targets["room_c"])
    return float(config.get("initial_guess", {}).get("room_c", -30.0))


def _set_proactive_control(
    config: dict[str, Any],
    *,
    enabled: bool,
    precool_lead_time_s: float,
    precool_depth_k: float,
    feedforward_enabled: bool,
    compressor_boost_fraction: float,
) -> None:
    proactive = dict(config.setdefault("control", {}).get("proactive", {}))
    proactive.update(
        {
            "enabled": bool(enabled),
            "precool_enabled": bool(enabled and precool_depth_k > 0.0 and precool_lead_time_s > 0.0),
            "precool_lead_time_s": float(precool_lead_time_s),
            "precool_depth_k": float(precool_depth_k),
            "feedforward_enabled": bool(enabled and feedforward_enabled and compressor_boost_fraction > 0.0),
            "compressor_boost_fraction": float(compressor_boost_fraction if feedforward_enabled else 0.0),
        }
    )
    config.setdefault("control", {})["proactive"] = proactive


def _set_first_door_open_time(config: dict[str, Any], first_door_open_s: float | None) -> None:
    if first_door_open_s is None:
        return

    infiltration_cfg = config.setdefault("disturbances", {}).setdefault("infiltration", {})
    start_s = float(first_door_open_s)
    schedule = infiltration_cfg.get("schedule")
    if isinstance(schedule, dict):
        if "events" in schedule and isinstance(schedule["events"], list):
            for event in schedule["events"]:
                if not isinstance(event, dict):
                    continue
                if "t_open_s" in event or "t_close_s" in event:
                    duration_s = float(
                        event.get(
                            "open_duration_s",
                            event.get("duration_s", float(event.get("t_close_s", start_s + 60.0)) - float(event.get("t_open_s", start_s))),
                        )
                    )
                    event["t_open_s"] = start_s
                    event["t_close_s"] = start_s + max(duration_s, 0.0)
                else:
                    event["start_time_s"] = start_s
                return

        if "t_open_s" in schedule or "t_close_s" in schedule:
            duration_s = float(
                schedule.get(
                    "open_duration_s",
                    schedule.get("duration_s", float(schedule.get("t_close_s", start_s + 60.0)) - float(schedule.get("t_open_s", start_s))),
                )
            )
            schedule["t_open_s"] = start_s
            schedule["t_close_s"] = start_s + max(duration_s, 0.0)
        else:
            schedule["start_time_s"] = start_s
        return

    if "t_open_s" in infiltration_cfg or "t_close_s" in infiltration_cfg:
        duration_s = float(
            infiltration_cfg.get(
                "open_duration_s",
                infiltration_cfg.get(
                    "duration_s",
                    float(infiltration_cfg.get("t_close_s", start_s + 60.0))
                    - float(infiltration_cfg.get("t_open_s", start_s)),
                ),
            )
        )
        infiltration_cfg["t_open_s"] = start_s
        infiltration_cfg["t_close_s"] = start_s + max(duration_s, 0.0)
    else:
        infiltration_cfg["start_time_s"] = start_s


def _apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    _set_first_door_open_time(config, args.first_door_open_s)
    if args.system_mode is not None:
        config.setdefault("system", {})["mode"] = args.system_mode
    if args.jacobian_workers is not None:
        config.setdefault("simulation", {})["jacobian_workers"] = max(1, int(args.jacobian_workers))
    fluids_cfg = config.setdefault("fluids", {})
    if args.property_backend is not None:
        fluids_cfg["property_backend"] = args.property_backend
    if args.property_backend == "refprop":
        fluids_cfg["refprop_path"] = args.refprop_path


def _series(history: list[dict[str, float]], key: str, default: float = np.nan) -> np.ndarray:
    return np.array([float(row.get(key, default)) for row in history], dtype=float)


def _integral_s(y: np.ndarray, t_s: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.trapezoid(np.nan_to_num(y, nan=0.0), t_s))


def _nanmin(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if values.size and not np.all(np.isnan(values)) else np.nan


def _nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if values.size and not np.all(np.isnan(values)) else np.nan


def _nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size and not np.all(np.isnan(values)) else np.nan


def _time_above_s(values: np.ndarray, threshold: float, t_s: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    above = np.where(values > threshold, 1.0, 0.0)
    return _integral_s(above, t_s)


def _time_to_recover_s(
    room_c: np.ndarray,
    t_s: np.ndarray,
    door_fraction: np.ndarray,
    threshold_c: float,
) -> float:
    if t_s.size == 0 or not np.any(door_fraction > 0.0):
        return 0.0
    last_open_index = int(np.max(np.nonzero(door_fraction > 0.0)[0]))
    for idx in range(last_open_index, room_c.size):
        if room_c[idx] <= threshold_c:
            return max(float(t_s[idx] - t_s[last_open_index]), 0.0)
    return float(t_s[-1] - t_s[last_open_index])


def _metrics(
    case_name: str,
    history: list[dict[str, float]],
    config: dict[str, Any],
    base_setpoint_c: float,
    critical_room_c: float,
) -> dict[str, float | str]:
    t_s = _series(history, "time_s", 0.0)
    room_c = _series(history, "room_c")
    dock_c = _series(history, "dock_c")
    power_w = _series(history, "w_total_input_w", 0.0)
    cop = _series(history, "cop_system")
    door_fraction = _series(history, "infiltration_door_open_fraction", 0.0)
    if np.all(np.isnan(door_fraction)):
        door_fraction = _series(history, "proactive_door_open_fraction", 0.0)

    infiltration_cfg = config.get("disturbances", {}).get("infiltration", {})
    door_starts = scheduled_door_event_starts_between(infiltration_cfg, float(t_s[0]), float(t_s[-1])) if t_s.size else []

    return {
        "case": case_name,
        "room_setpoint_c": base_setpoint_c,
        "critical_room_c": critical_room_c,
        "door_events": len(door_starts),
        "min_room_c": _nanmin(room_c),
        "max_room_c": _nanmax(room_c),
        "max_room_excursion_k": _nanmax(room_c - base_setpoint_c),
        "max_room_above_critical_k": max(_nanmax(room_c - critical_room_c), 0.0),
        "time_room_above_critical_s": _time_above_s(room_c, critical_room_c, t_s),
        "recovery_time_after_last_open_s": _time_to_recover_s(room_c, t_s, door_fraction, critical_room_c),
        "max_dock_c": _nanmax(dock_c),
        "energy_kwh": _integral_s(power_w, t_s) / 3.6e6,
        "peak_power_kw": _nanmax(power_w) / 1000.0,
        "mean_cop": _nanmean(cop),
        "max_refrigerant_compressor_speed_rpm": _nanmax(_series(history, "refrigerant_compressor_speed_rpm")),
    }


def _add_baseline_deltas(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    baseline = rows[0]
    for row in rows:
        row["max_room_reduction_vs_baseline_k"] = float(baseline["max_room_c"]) - float(row["max_room_c"])
        row["energy_delta_vs_baseline_kwh"] = float(row["energy_kwh"]) - float(baseline["energy_kwh"])
        row["peak_power_delta_vs_baseline_kw"] = float(row["peak_power_kw"]) - float(baseline["peak_power_kw"])
        row["critical_time_reduction_vs_baseline_s"] = (
            float(baseline["time_room_above_critical_s"]) - float(row["time_room_above_critical_s"])
        )


def _write_summary(rows: list[dict[str, float | str]], path: Path) -> None:
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


def _plot_comparison(
    histories: dict[str, list[dict[str, float]]],
    output_path: Path,
    room_setpoint_c: float,
    critical_room_c: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    for case_name, history in histories.items():
        t_min = _series(history, "time_s", 0.0) / 60.0
        axes[0].plot(t_min, _series(history, "room_c"), label=case_name)
        axes[1].plot(t_min, _series(history, "infiltration_door_open_fraction", 0.0), label=case_name)
        axes[2].plot(t_min, _series(history, "w_total_input_w", 0.0) / 1000.0, label=case_name)
        axes[3].plot(t_min, _series(history, "cop_system"), label=case_name)

    axes[0].axhline(room_setpoint_c, color="black", linestyle=":", linewidth=1.0, label="setpoint")
    axes[0].axhline(critical_room_c, color="tab:red", linestyle="--", linewidth=1.0, label="critical")
    axes[0].set_ylabel("Room [C]")
    axes[1].set_ylabel("Door open [-]")
    axes[2].set_ylabel("Power [kW]")
    axes[3].set_ylabel("COP [-]")
    axes[3].set_xlabel("Time [min]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_control_study(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_label or Path(args.config).stem
    base_setpoint_c = _room_setpoint_c(base_config)
    critical_room_c = args.critical_room_c
    if critical_room_c is None:
        critical_room_c = base_setpoint_c + args.allowed_excursion_k

    case_specs = [
        ("baseline_feedback", False, 0.0, False, 0.0),
        ("scheduled_precooling", True, args.precool_depth_k, False, 0.0),
    ]
    if not args.no_feedforward and args.boost_fraction > 0.0:
        case_specs.append(("precooling_feedforward", True, args.precool_depth_k, True, args.boost_fraction))

    histories: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, float | str]] = []
    for case_name, enabled, depth_k, feedforward_enabled, boost_fraction in case_specs:
        case_config = _with_case_output(base_config, output_dir, run_label, case_name)
        _set_proactive_control(
            case_config,
            enabled=enabled,
            precool_lead_time_s=args.precool_lead_time_s,
            precool_depth_k=depth_k,
            feedforward_enabled=feedforward_enabled,
            compressor_boost_fraction=boost_fraction,
        )
        _apply_runtime_overrides(case_config, args)
        history = run_simulation(case_config)
        histories[case_name] = history
        save_csv(history, case_config["output"]["csv_file"])
        rows.append(_metrics(case_name, history, case_config, base_setpoint_c, critical_room_c))

    _add_baseline_deltas(rows)
    summary_path = output_dir / f"{_safe_label(run_label)}_control_study_summary.csv"
    plot_path = output_dir / f"{_safe_label(run_label)}_control_study.png"
    _write_summary(rows, summary_path)
    _plot_comparison(histories, plot_path, base_setpoint_c, critical_room_c)

    print(f"Study summary saved to {summary_path.resolve()}")
    print(f"Comparison plot saved to {plot_path.resolve()}")
    for row in rows:
        print(
            f"[{row['case']}] max_room={float(row['max_room_c']):.2f} C, "
            f"energy={float(row['energy_kwh']):.3f} kWh, "
            f"peak_power={float(row['peak_power_kw']):.2f} kW, "
            f"critical_time={float(row['time_room_above_critical_s']):.1f} s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare feedback, scheduled precooling, and feedforward control.")
    parser.add_argument("--config", required=True, help="Path to the base JSON configuration.")
    parser.add_argument("--output-dir", default="outputs/proactive_control", help="Directory for study CSV and plots.")
    parser.add_argument("--run-label", help="Optional label used in output filenames.")
    parser.add_argument("--precool-lead-time-s", type=float, default=900.0, help="Precooling lead time before door opening.")
    parser.add_argument("--precool-depth-k", type=float, default=3.0, help="Room setpoint reduction during scheduled precooling.")
    parser.add_argument("--boost-fraction", type=float, default=0.5, help="Door-open compressor feedforward boost fraction.")
    parser.add_argument("--first-door-open-s", type=float, help="Override the first scheduled door opening time in seconds.")
    parser.add_argument("--no-feedforward", action="store_true", help="Only compare baseline against scheduled precooling.")
    parser.add_argument("--system-mode", choices=("cascade", "air_cycle", "vcc"), help="Override the config system mode.")
    parser.add_argument("--jacobian-workers", type=int, help="Thread workers per simulation for finite-difference Jacobians.")
    parser.add_argument(
        "--property-backend",
        choices=("coolprop", "refprop"),
        help="Thermophysical property backend override. Leave unset to use the config/default.",
    )
    parser.add_argument(
        "--refprop-path",
        default=DEFAULT_REFPROP_PATH,
        help=f"REFPROP installation path used with --property-backend refprop. Default: {DEFAULT_REFPROP_PATH}",
    )
    parser.add_argument(
        "--allowed-excursion-k",
        type=float,
        default=2.0,
        help="Critical room threshold above the base setpoint when --critical-room-c is not supplied.",
    )
    parser.add_argument("--critical-room-c", type=float, help="Absolute critical room temperature threshold in Celsius.")
    run_control_study(parser.parse_args())


if __name__ == "__main__":
    main()
