from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from .batch import CaseResult, apply_output_version, run_case_from_config_path, run_cases
from .config import load_config
from .fluids import DEFAULT_REFPROP_PATH
from .simulation import run_simulation, save_csv, save_plot


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _cycle_cop(numerator_w: float, denominator_w: float, reported: object) -> float | None:
    reported_cop = _finite_float(reported)
    if reported_cop is not None:
        return reported_cop
    if abs(denominator_w) <= 1.0e-9:
        return None
    return numerator_w / denominator_w


def _format_optional(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _print_cycle_summary_table(values: dict[str, object]) -> None:
    q_room_w = _finite_float(values.get("q_room_w"))
    w_air_input_w = _finite_float(values.get("w_air_input_w"))
    m_air_kg_s = _finite_float(values.get("m_air_kg_s"))
    air_pressure_ratio = _finite_float(values.get("air_pressure_ratio"))

    q_cascade_w = _finite_float(values.get("q_cascade_w"))
    q_dock_w = _finite_float(values.get("q_dock_w"))
    w_ref_comp_w = _finite_float(values.get("w_ref_comp_w"))
    m_ref_kg_s = _finite_float(values.get("m_ref_kg_s"))
    refrigerant_pressure_ratio = _finite_float(values.get("refrigerant_pressure_ratio"))

    rows: list[list[str]] = []
    if q_room_w is not None and w_air_input_w is not None:
        cop_air = _cycle_cop(q_room_w, w_air_input_w, values.get("cop_air_cycle"))
        rows.append(
            [
                "Air cycle",
                "Q_room",
                f"{q_room_w / 1000.0:.3f}",
                "W_net",
                f"{w_air_input_w / 1000.0:.3f}",
                _format_optional(m_air_kg_s, 4),
                _format_optional(air_pressure_ratio, 3),
                _format_optional(cop_air, 3),
            ]
        )

    if q_cascade_w is not None and q_dock_w is not None and w_ref_comp_w is not None:
        vcc_load_w = q_cascade_w + q_dock_w
        cop_vcc = _cycle_cop(vcc_load_w, w_ref_comp_w, values.get("cop_vcc"))
        rows.append(
            [
                "VCC",
                "Q_cascade + Q_dock",
                f"{vcc_load_w / 1000.0:.3f}",
                "W_NH3_comp",
                f"{w_ref_comp_w / 1000.0:.3f}",
                _format_optional(m_ref_kg_s, 4),
                _format_optional(refrigerant_pressure_ratio, 3),
                _format_optional(cop_vcc, 3),
            ]
        )

    if not rows:
        return

    headers = ["Cycle", "Cooling basis", "Cooling [kW]", "Input basis", "Input [kW]", "m_dot [kg/s]", "PR", "COP"]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def line(cells: list[str]) -> str:
        return "  " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print("Cycle performance summary")
    print(line(headers))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def _print_final_state(last: dict[str, float], plot_file: str | Path, csv_file: str | Path) -> None:
    def maybe(label: str, key: str, scale: float = 1.0, suffix: str = "") -> None:
        value = last.get(key)
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if numeric != numeric:
            return
        print(f"  {label:<22}: {numeric / scale:.4g}{suffix}")

    def maybe_temperature(label: str, kelvin_key: str, celsius_key: str) -> None:
        if kelvin_key in last:
            maybe(label, kelvin_key, suffix=" K")
        else:
            maybe(label, celsius_key, suffix=" C")

    print("Final dynamic state")
    maybe_temperature("Refrigerating temp", "refrigerating_temperature_k", "refrigerating_temperature_c")
    maybe_temperature("Room temperature", "room_k", "room_c")
    maybe_temperature("Dock temperature", "dock_k", "dock_c")
    maybe_temperature("Sink temperature", "sink_k", "sink_c")
    maybe_temperature("Water loop temp", "water_loop_k", "water_loop_c")
    maybe_temperature("Evaporating temp", "tevap_k", "tevap_c")
    maybe_temperature("Condensing temp", "tcond_k", "tcond_c")
    maybe("Cooling capacity", "q_room_w", scale=1000.0, suffix=" kW")
    maybe("Dock evaporator duty", "q_dock_w", scale=1000.0, suffix=" kW")
    maybe("Useful cooling total", "q_useful_w", scale=1000.0, suffix=" kW")
    maybe("Cascade duty", "q_cascade_w", scale=1000.0, suffix=" kW")
    maybe("Condenser/reject duty", "q_cond_w", scale=1000.0, suffix=" kW")
    maybe("NH3 compressor work", "w_ref_comp_w", scale=1000.0, suffix=" kW")
    maybe("NH3 superheat", "refrigerant_superheat_k", suffix=" K")
    maybe("NH3 isentropic work", "w_ref_isentropic_w", scale=1000.0, suffix=" kW")
    maybe("Air input power", "w_air_input_w", scale=1000.0, suffix=" kW")
    maybe("System COP", "cop_system")
    maybe("Room-only COP", "cop_room_only")
    maybe("Dry-air massflow", "m_air_kg_s", suffix=" kg/s")
    maybe("Ice at air outlet", "ice_mass_flow_kg_s", suffix=" kg/s")
    maybe("Refrigerant massflow", "m_ref_kg_s", suffix=" kg/s")
    maybe("Receiver mass", "receiver_mass_kg", suffix=" kg")
    maybe("Receiver fill", "receiver_liquid_fill_fraction")
    _print_cycle_summary_table(last)
    print(f"  Plot saved to        : {Path(plot_file).resolve()}")
    print(f"  CSV saved to         : {Path(csv_file).resolve()}")


def _case_result_final_values(result: CaseResult) -> dict[str, object]:
    return {
        "q_room_w": result.final_q_room_w,
        "q_cascade_w": result.final_q_cascade_w,
        "q_dock_w": result.final_q_dock_w,
        "w_air_input_w": result.final_w_air_input_w,
        "w_ref_comp_w": result.final_w_ref_comp_w,
        "cop_air_cycle": result.final_cop_air_cycle,
        "cop_vcc": result.final_cop_vcc,
        "m_air_kg_s": result.final_m_air_kg_s,
        "m_ref_kg_s": result.final_m_ref_kg_s,
        "air_pressure_ratio": result.final_air_pressure_ratio,
        "refrigerant_pressure_ratio": result.final_refrigerant_pressure_ratio,
    }


def _print_case_result(result: CaseResult) -> None:
    duration = "base" if result.door_open_duration_s is None else f"{result.door_open_duration_s:g} s"
    print(f"[{result.name}] door open duration: {duration}")
    print(f"  Room temperature     : {result.final_room_c:.2f} C")
    print(f"  Dock temperature     : {result.final_dock_c:.2f} C")
    print(f"  Sink temperature     : {result.final_sink_c:.2f} C")
    print(f"  System COP           : {result.final_cop_system:.3f}")
    print(f"  Dry-air massflow     : {result.final_m_air_kg_s:.4f} kg/s")
    print(f"  Refrigerant massflow : {result.final_m_ref_kg_s:.4f} kg/s")
    _print_cycle_summary_table(_case_result_final_values(result))
    print(f"  Plot saved to        : {result.plot_file}")
    print(f"  CSV saved to         : {result.csv_file}")


def _door_durations_from_args(args: argparse.Namespace) -> list[float] | None:
    durations: list[float] = []
    if args.door_open_duration is not None:
        durations.extend(args.door_open_duration)
    if args.door_open_durations is not None:
        durations.extend(args.door_open_durations)
    return durations or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic simulation of a cascade reverse-Brayton / VCC refrigeration system.")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file.")
    parser.add_argument(
        "--door-open-duration",
        type=float,
        action="append",
        help="Run one case with this door open duration in seconds. Can be repeated.",
    )
    parser.add_argument(
        "--door-open-durations",
        type=float,
        nargs="+",
        help="Run one or more cases with these door open durations in seconds, e.g. 30 60 120.",
    )
    parser.add_argument("--parallel", action="store_true", help="Run multiple requested cases in parallel processes.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for --parallel.")
    parser.add_argument(
        "--jacobian-workers",
        type=int,
        default=None,
        help="Thread workers per simulation for finite-difference Jacobian evaluations.",
    )
    parser.add_argument(
        "--run-version",
        help="Append a stable version tag to output files, e.g. 2 creates *_2.csv and rerunning 2 overwrites it.",
    )
    parser.add_argument(
        "--system-mode",
        choices=("cascade", "air_cycle", "vcc"),
        help="Override the config system mode. Use air_cycle or vcc for standalone validation runs.",
    )
    parser.add_argument(
        "--property-backend",
        choices=("coolprop", "refprop"),
        default="refprop",
        help="Thermophysical property backend for the refrigerant cycle. Default: refprop.",
    )
    parser.add_argument(
        "--refprop-path",
        default=DEFAULT_REFPROP_PATH,
        help=f"REFPROP installation path used when --property-backend refprop. Default: {DEFAULT_REFPROP_PATH}",
    )
    args = parser.parse_args()

    door_durations = _door_durations_from_args(args)
    if door_durations is None:
        cfg = load_config(args.config)
        if args.system_mode is not None:
            cfg.setdefault("system", {})["mode"] = args.system_mode
        if args.jacobian_workers is not None:
            cfg["simulation"]["jacobian_workers"] = max(1, args.jacobian_workers)
        cfg.setdefault("fluids", {})["property_backend"] = args.property_backend
        if args.property_backend == "refprop":
            cfg["fluids"]["refprop_path"] = args.refprop_path
        apply_output_version(cfg, args.run_version)
        history = run_simulation(cfg)
        save_plot(history, cfg["output"]["plot_file"])
        save_csv(history, cfg["output"]["csv_file"])
        _print_final_state(history[-1], cfg["output"]["plot_file"], cfg["output"]["csv_file"])
        return

    if len(door_durations) == 1:
        result = run_case_from_config_path(
            args.config,
            door_durations[0],
            args.run_version,
            args.jacobian_workers,
            args.property_backend,
            args.refprop_path if args.property_backend == "refprop" else None,
            args.system_mode,
        )
        _print_case_result(result)
        return

    workers = 1
    if args.parallel:
        workers = args.workers or min(len(door_durations), os.cpu_count() or 1)
    elif args.workers is not None and args.workers > 1:
        workers = args.workers

    print(f"Running {len(door_durations)} door-duration cases with workers={workers}")
    results = run_cases(
        args.config,
        list(door_durations),
        workers=workers,
        run_version=args.run_version,
        jacobian_workers=args.jacobian_workers,
        property_backend=args.property_backend,
        refprop_path=args.refprop_path if args.property_backend == "refprop" else None,
        system_mode=args.system_mode,
    )
    for result in results:
        _print_case_result(result)


if __name__ == "__main__":
    main()
