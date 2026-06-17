from __future__ import annotations

import argparse
import os
from pathlib import Path

from .batch import CaseResult, apply_output_version, run_case_from_config_path, run_cases
from .config import load_config
from .fluids import DEFAULT_REFPROP_PATH
from .simulation import run_simulation, save_csv, save_plot


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
    print(f"  Plot saved to        : {Path(plot_file).resolve()}")
    print(f"  CSV saved to         : {Path(csv_file).resolve()}")


def _print_case_result(result: CaseResult) -> None:
    duration = "base" if result.door_open_duration_s is None else f"{result.door_open_duration_s:g} s"
    print(f"[{result.name}] door open duration: {duration}")
    print(f"  Room temperature     : {result.final_room_c:.2f} C")
    print(f"  Dock temperature     : {result.final_dock_c:.2f} C")
    print(f"  Sink temperature     : {result.final_sink_c:.2f} C")
    print(f"  System COP           : {result.final_cop_system:.3f}")
    print(f"  Dry-air massflow     : {result.final_m_air_kg_s:.4f} kg/s")
    print(f"  Refrigerant massflow : {result.final_m_ref_kg_s:.4f} kg/s")
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
