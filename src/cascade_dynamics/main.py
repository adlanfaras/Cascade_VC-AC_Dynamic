from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .simulation import run_simulation, save_csv, save_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic simulation of a cascade reverse-Brayton / VCC refrigeration system.")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    history = run_simulation(cfg)
    save_plot(history, cfg["output"]["plot_file"])
    save_csv(history, cfg["output"]["csv_file"])

    last = history[-1]
    print("Final dynamic state")
    print(f"  Room temperature     : {last['room_c']:.2f} C")
    print(f"  Dock temperature     : {last['dock_c']:.2f} C")
    print(f"  Sink temperature     : {last['sink_c']:.2f} C")
    print(f"  Cooling capacity     : {last['q_room_w'] / 1000.0:.2f} kW")
    print(f"  Dock evaporator duty : {last['q_dock_w'] / 1000.0:.2f} kW")
    print(f"  Useful cooling total : {last['q_useful_w'] / 1000.0:.2f} kW")
    print(f"  Cascade duty         : {last['q_cascade_w'] / 1000.0:.2f} kW")
    print(f"  Condenser duty       : {last['q_cond_w'] / 1000.0:.2f} kW")
    print(f"  NH3 compressor work  : {last['w_ref_comp_w'] / 1000.0:.2f} kW")
    print(f"  NH3 superheat        : {last['refrigerant_superheat_k']:.2f} K")
    print(f"  NH3 isentropic work  : {last['w_ref_isentropic_w'] / 1000.0:.2f} kW")
    print(f"  Air input power      : {last['w_air_input_w'] / 1000.0:.2f} kW")
    print(f"  System COP           : {last['cop_system']:.3f}")
    print(f"  Room-only COP        : {last['cop_room_only']:.3f}")
    print(f"  Dry-air massflow     : {last['m_air_kg_s']:.4f} kg/s")
    print(f"  Ice at air outlet    : {last['ice_mass_flow_kg_s']:.4f} kg/s")
    print(f"  Refrigerant massflow : {last['m_ref_kg_s']:.4f} kg/s")
    print(f"  Plot saved to        : {Path(cfg['output']['plot_file']).resolve()}")
    print(f"  CSV saved to         : {Path(cfg['output']['csv_file']).resolve()}")


if __name__ == "__main__":
    main()
