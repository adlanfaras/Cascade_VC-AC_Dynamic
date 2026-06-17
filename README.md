# Cascade Refrigeration Dynamic Simulator

Dynamic simulator for a cascade refrigeration system with:

- reverse-Brayton air cycle on the cold side
- ammonia vapor-compression cycle on the warm side
- loading-dock evaporator branch
- optional door-opening infiltration disturbance cases

The simulator writes CSV results and plots to `outputs/`.

## Standalone Validation Modes

The default `cascade` mode still solves the coupled reverse-Brayton / VCC system. For component-level validation, set `system.mode` in the config or pass `--system-mode`:

```json
"system": {
  "mode": "air_cycle"
}
```

Supported modes:

```text
cascade    Coupled air-cycle + VCC model.
air_cycle  Reverse-Brayton air cycle only, with the cascade cooler replaced by a water loop.
vcc        Ammonia vapor-compression cycle only, with fixed evaporator loads.
```

Air-cycle standalone mode uses `air_cycle.water_loop` to define the water-cooled cascade exchanger and the hot-side air cooler:

```json
"air_cycle": {
  "water_loop": {
    "initial_c": 33.0,
    "capacitance_j_k": 500000.0,
    "cascade_ua_w_k": 8000.0,
    "air_cooler_ua_w_k": 18000.0,
    "ambient_c": 30.0
  }
}
```

VCC standalone mode uses fixed validation loads:

```json
"vcc_cycle": {
  "standalone": {
    "evaporator_loads": {
      "cascade_w": 43845.0,
      "dock_w": 25750.0
    }
  }
}
```

The ammonia VCC can include an optional high-pressure liquid receiver between the condenser and expansion valves:

```json
"vcc_cycle": {
  "receiver": {
    "enabled": true,
    "initial_residence_time_s": 5.0,
    "initial_liquid_fill_fraction": 0.5,
    "inlet_time_constant_s": 5.0
  }
}
```

When `volume_m3` or `initial_mass_kg` is omitted, the simulator auto-sizes the startup inventory from `initial_guess.m_ref_kg_s`, the residence time, and the requested initial fill fraction. Receiver diagnostics are written to CSV as `receiver_mass_kg`, `receiver_liquid_fill_fraction`, `receiver_inlet_m_dot_kg_s`, and `receiver_outlet_m_dot_kg_s`.

Template configs are included:

```powershell
python -m src.cascade_dynamics.main --config config/validation_air_cycle_water_loop.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/validation_vcc_fixed_load.json --property-backend coolprop
```

## Salgado Thesis VCC Validation

The isolated VCC mode includes direct Salgado thesis validation cases for the three warm-side layouts:

```powershell
# Layout A: high-pressure receiver, Table 3.4 frequency sweep
python -m src.cascade_dynamics.main --config config/salgado_layout_a_hpr_25hz.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/salgado_layout_a_hpr_30hz.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/salgado_layout_a_hpr_35hz.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/salgado_layout_a_hpr_40hz.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/salgado_layout_a_hpr_43hz.json --property-backend coolprop

# Layout B: condenser subcooler, Tables 3.11-3.14
python -m src.cascade_dynamics.main --config config/salgado_layout_b_csc_charge_1p88kg.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/salgado_layout_b_csc_charge_2p02kg.json --property-backend coolprop

# Layout C: low-pressure receiver concepts, Tables 4.1-4.7 and Appendix C
python -m src.cascade_dynamics.main --config config/salgado_layout_c_lpr_concept1_bitzer.json --property-backend coolprop
python -m src.cascade_dynamics.main --config config/salgado_layout_c_lpr_concept2_gea.json --property-backend coolprop
```

Layout behavior is selected with `vcc_cycle.layout`:

```json
{
  "vcc_cycle": {
    "layout": "hpr",
    "subcooling_k": 0.0,
    "receiver": {
      "enabled": true,
      "force_saturated_liquid_outlet": true
    }
  }
}
```

- `hpr` forces the valve inlet to saturated liquid at condenser pressure.
- `csc` uses the configured condenser outlet subcooling and reports `refrigerant_q_subcooler_w`.
- `lpr` forces compressor suction to saturated vapor and should use subcooling control on the EEV instead of superheat control.

The LPR concept configs use Appendix C pressure-ratio polynomials through:

```json
{
  "compressor": {
    "model": "screw_pressure_ratio_polynomial",
    "preset": "bitzer_osha7462_k"
  }
}
```

Available presets are `bitzer_osha7462_k` and `gea_eb_7a`; configs may also provide `eta_is_coefficients` and `eta_v_coefficients` directly. The current isolated VCC model still represents heat exchangers as lumped UA components. Yang et al. single-phase PHE heat-transfer correlations therefore require a future finite-zone PHE model before they can replace water-side or superheated/subcooled-zone correlations without bypassing the existing UA formulation.

## Dynamic Heat-Exchanger UA

Heat-exchanger conductance is evaluated with Model A single-stream scaling:

```text
UA = UA_nominal * (m_dot / m_dot_nominal)^n
```

The existing `*_ua_w_k` config values remain the nominal design conductances. Effective runtime UA values are written to CSV as `regenerator_ua_w_k`, `cascade_ua_w_k`, `condenser_ua_w_k`, and `dock_evaporator_ua_w_k`; the matching `*_ua_ref_w_k` fields hold the nominal values. Default exponents are:

```json
{
  "regenerator": 0.8,
  "cascade": 0.8,
  "condenser": 0.8,
  "dock_evaporator": 0.6,
  "water_loop_air_cooler": 0.8
}
```

Nominal flows are inferred from each case at startup (`fixed_m_dot_kg_s`, `sink_m_dot_kg_s`, and dock `design_air_m_dot_kg_s`). To override or disable scaling, add a top-level block such as:

```json
"heat_exchanger_ua_scaling": {
  "enabled": true,
  "min_flow_ratio": 0.0,
  "cascade": {
    "exponent": 0.8,
    "nominal_m_dot_kg_s": 3.35
  }
}
```

## Setup

```powershell
pip install -r requirements.txt
```

## Configs

Use one of the included refrigerated-space or loading-dock infiltration configs:

```text
config/RS_Infiltration_Qvc100_Qld25_2.json
config/RS_Infiltration_Qvc100_Qld25_4.json
config/RS_Infiltration_Qvc100_Qld25_8.json
config/LD_Infiltration_Qvc100_Qld25_1.json
```

The refrigerated-space (`RS_...`) matrix covers matching Qvc/Qld load variations with `2`, `4`, and `8` openings per hour. The loading-dock (`LD_...`) matrix covers the same Qvc/Qld load variations with one opening per hour only. Loading-dock configs apply ambient infiltration to the dock load only. All included infiltration configs use an effectiveness of `0.96` and a `2.8 m x 3.3 m` door.

Door-opening disturbances can be configured with explicit open/close times, or with a repeating start/duration/interval schedule. Durations may be provided in seconds or hours:

```json
"infiltration": {
  "enabled": true,
  "model": "tian_unsteady",
  "schedule": {
    "start_time_s": 5.0,
    "ramp_time_s": 2.0,
    "open_duration_s": 12.0,
    "interval_s": 450.0,
    "repeat_count": 8,
    "opening_fraction": 1.0
  },
  "effectiveness": 0.96,
  "door": {
    "width_m": 2.8,
    "height_m": 3.3
  }
}
```

Here `interval_s` is the time from one opening start to the next opening start. RS configs use `1800 s`, `900 s`, and `450 s` intervals for the `_2`, `_4`, and `_8` files. LD configs use `3600 s` and `repeat_count: 1`. Use `door_ramp_open_time_s` and `door_ramp_close_time_s` if the opening and closing ramps should have different durations. Existing `t_open_s`/`t_close_s` configs remain supported.

## Run One Case

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json
```

```powershell
python -m src.cascade_dynamics.main --config config/LD_Infiltration_Qvc100_Qld25_1.json
```

## Property Backend

The CLI uses REFPROP by default with `C:\Program Files (x86)\REFPROP`:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json
```

You can also pass REFPROP explicitly:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --property-backend refprop
```

The default REFPROP path is `C:\Program Files (x86)\REFPROP`. Override it if needed:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --property-backend refprop --refprop-path "C:\Program Files (x86)\REFPROP"
```

Use CoolProp instead with:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --property-backend coolprop
```

## Run Door-Duration Cases

Run one modified door-open duration:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-duration 30
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-duration 30
```

Run 30 s, 60 s, and 120 s cases:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120
```

## Parallel Runs

Run multiple door-duration cases in parallel:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120 --parallel --workers 3
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120 --parallel --workers 3
```

With only three door-duration cases, the outer batch can only keep three processes busy. You can also experiment with parallel finite-difference Jacobian work inside each simulation:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120 --parallel --workers 3 --jacobian-workers 3
```

Benchmark `--jacobian-workers 2`, `3`, or `4` before using it for production runs. Very high values can be slower because each timestep solves a small 10-variable system.

## Versioned Outputs

Add a stable version tag to output filenames:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --run-version 2
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --run-version 2
```

Combine versioned outputs with parallel door-duration runs:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120 --parallel --workers 3 --run-version 2
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120 --parallel --workers 3 --run-version 2
```

With `--run-version 2`, outputs are written with suffixes such as:

```text
outputs/RS_Infiltration_Qvc100_Qld25_8_door_30s_2.csv
outputs/RS_Infiltration_Qvc100_Qld25_8_door_30s_2.png
```

Rerunning the same version overwrites that version's files.
