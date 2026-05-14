# Cascade Refrigeration Dynamic Simulator

This project simulates the dynamic performance of a cascade refrigeration system made of:

- a bottom reverse Brayton air cycle
- a top vapor-compression ammonia cycle
- an explicit loading-dock evaporator branch on the ammonia side

The implementation uses:

- `CoolProp` for thermophysical properties
- `numpy` for array math
- `matplotlib` for plots
- a custom Newton-Raphson nonlinear solver with finite-difference Jacobian

## Model scope

The model is organized as a semi-explicit DAE solved with backward Euler in time:

- Dynamic states:
  - refrigerated room / compressor inlet temperature
  - loading-dock air temperature
  - condenser sink outlet / lumped sink temperature
- Algebraic variables:
  - air-cycle regenerator outlet temperatures
  - air temperature entering the turbine
  - air supply temperature to the cold room
  - ammonia evaporating temperature
  - ammonia mass flow through the expansion valve

Component models:

- compressor / turbine: isentropic efficiency
- bottom-side air properties: saturated humid air in the refrigerated space,
  constant absolute humidity until turbine expansion, and equilibrium
  vapour-to-ice formation during expansion
- regenerator / cascade HX / condenser: `UA-LMTD`
- room load: heater model, `Q_RS = m_air * (h_room_return - h_supply)`
- loading dock branch: lumped dock air temperature with `UA * (T_dock - T_evap)`
  evaporator heat removal
- expansion valve: isenthalpic with linear pressure-drop flow equation
- bottom-side air flow: compressor head-to-volumetric-flow polynomial with
  suction-density conversion to mass flow, driven by compressor head only
- ammonia compressor: variable-speed BITZER map using evaporating temperature,
  condensing temperature, and compressor speed to predict refrigerant mass flow
  and shaft power

The bottom-side air-cycle layout used by the model is:

```text
6 -> compressor -> 2 -> cascade HX -> 3 -> regenerator hot side -> 4
4 -> turbine -> 5 -> refrigerated room heater/load -> room_c
room_c -> regenerator cold side -> 6

NH3 evaporator load:

```text
Q_evap_total = Q_cascade + Q_LD
```
```

So the cascade HX cools the compressor discharge before the regenerator, and
the regenerator LMTD is based on:

```text
LMTD_reg = LMTD(T3 - T6, T4 - Troom)
```

The bottom-side air mass flow is computed from the compressor isentropic head.
The humid-air properties are expressed per kg of dry air, so `m_air` is the dry
air mass flow:

```text
h_is = J(P2, s1, x_room) - J1
H = h_is / g
Q = f(H, rpm)
m_air = rho_dry_air,suction * Q
```

For the expander, the model follows the paper's humid expansion treatment:

```text
x_room = x_sat(T_room, P_low) * RH_room
J = h_air + x_v h_v + x_ice h_ice
s = s_air + x_v s_v + x_ice s_ice
x_v = min(x_room, x_sat(T, P))
x_ice = max(x_room - x_v, 0)
eta_T = (J4 - J5) / (J4 - J5,is)
```

The active compressor map expects:

```text
H in meters of isentropic head
rpm in compressor speed
Q in m^3/s
```

The air-cycle controller acts on
`air_cycle.compressor_mass_flow.speed_rpm`, using the air temperature before
the room (`t5_c`) as its measurement.

For the ammonia compressor, the configured reference case now uses a variable
speed map:

```text
mdot_ref = map(tevap, tcond, speed_rpm)
Wcomp_ref = map(tevap, tcond, speed_rpm)
Qcond = Qevap_total + Wcomp_ref
```

The solver treats `tcond_c` as an algebraic variable, gets condensing pressure
from `Psat(tcond)`, and matches the compressor-map mass flow against the
expansion-valve flow relation.

## Run

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json
```

The script prints a short summary and writes plots into `outputs/`.

Before the transient run starts, `simulation.startup_initialization` can run a
separate startup design-point solve. This mode does not advance time. It writes
startup values back into the in-memory config, then freezes them for the dynamic
time loop. If `freeze_solved_parameters` is true, controllers whose actuator
path was solved during startup report the frozen value instead of overwriting it.

The reference case uses `mode: "paper_design_point"`. In this mode the
bottom-cycle air temperatures are constructed from the paper equations:

```text
T5 = TRS - DT5-6
T3 = Tevap + DTeva,min
T4 = T3 - eR * (T3 - TRS)
T1 = TRS + eR * (T3 - TRS)
```

In this codebase, `room_c` corresponds to `TRS`, and `t6_c` corresponds to the
compressor inlet `T1`. The initializer then solves the pressure ratio that gives
the target humid-air turbine outlet temperature, solves the air-compressor speed
to match the paper top-cycle cooling capacity (`Qf-VC = 103 kW` in the
reference case), and back-calculates the room load, heat-exchanger UAs, sink
flow, refrigerant mass flow, and expansion-valve opening so the coupled model
starts from that paper-like state. With `solve_refrigerant_compressor_speed:
false`, the NH3 compressor remains at the configured reference speed and its
map mass flow is used.

The startup problem is configured by:

- `targets`: design-point temperatures for room, loading dock, sink,
  evaporator, and condenser
- optional targets such as `t5_target_c`, `room_delta_t_target_c`, and
  `superheat_target_k`
- paper-mode targets such as `cascade_air_evap_min_delta_t_c`,
  `regenerator_effectiveness`, and `target_vcc_cooling_capacity_w`
- `free_state_variables`: internal algebraic temperatures that may move while
  fitting the operating point, along with free refrigerant mass flow if it is
  not listed as a target
- `free_parameters`: config paths that may be solved during startup, such as
  condenser UA, cascade HX UA, expansion-valve opening, NH3 compressor speed,
  air-cycle pressure ratio, or air-compressor speed

Startup solved parameters are also copied into the first CSV row as
`startup_solved_*` snapshot columns. For active controllers, the solved actuator
value is also copied into the controller bias before the transient loop starts,
so the first controlled step does not jump away from the initialized operating
point. Disable startup initialization with:

```json
"startup_initialization": {
  "enabled": false
}
```

Temperature inputs and outputs in the JSON config, CSV, plots, and printed
summary use degrees Celsius. The thermodynamic property calls convert Celsius to
Kelvin internally only where CoolProp requires absolute temperature.

## Disturbance Input

Infiltration is configured in JSON under `disturbances.infiltration`.

The preferred door-opening model is the Tian-style unsteady analytical model:

```json
{
  "enabled": true,
  "model": "tian_unsteady",
  "t_open_s": 60.0,
  "t_close_s": 140.0,
  "outdoor_source": "dock",
  "outdoor_relative_humidity": 0.65,
  "effectiveness": 0.0,
  "door": {
    "width_m": 1.2,
    "height_m": 2.2
  },
  "room": {
    "width_m": 6.0,
    "length_m": 10.0,
    "height_m": 3.0
  }
}
```

This mode integrates three door-event states:

```text
v(t)   = representative infiltration velocity [m/s]
rho(t) = average air density in the door infiltration region [kg/m^3]
I(t)   = cumulative infiltrated volume [m^3]
```

The airflow reported to CSV is:

```text
Q_inf = v * A_flow
A_flow = W_d * H_d / 2
```

During an open event, the model uses buoyancy pressure, door pressure, and
friction/local resistance terms to generate the expected transient pulse:
rapid rise, peak, then gradual decay while the door remains open. The cumulative
volume switches the lower outgoing stream from initial cold room air to the
mixed infiltration-region density after `I >= V_eff`. Set
`stage_smoothing_m3` to a positive value to smooth that switch.

The room receives:

```text
q_total = Q_inf * rho_o * Cp_air * (T_o - T_room)
        + Q_inf * rho_o * max(omega_o - omega_room, 0) * h_latent
```

and the dock receives the equal-and-opposite load, preserving the coupled
room-to-dock exchange logic. With `outdoor_source: "dock"`, `T_o` follows the
dynamic dock temperature. Use `outdoor_source: "ambient"` for the configured
ambient temperature or `outdoor_source: "fixed"` with `outdoor_c`.

The effective infiltration length defaults to the conservative fallback
`L_el = L_c`. If a calibrated or paper-specific maximum is available, set
`L_max_m` or directly set `effective_length_m`.

CSV output includes:

```text
infiltration_q_m3_s
infiltration_sensible_w
infiltration_latent_w
infiltration_total_w
infiltration_cumulative_volume_m3
infiltration_velocity_m_s
infiltration_region_density_kg_m3
infiltration_stage
```

The older fixed-magnitude pulse is still available:

```json
{
  "enabled": true,
  "start_time_s": 900.0,
  "delay_s": 120.0,
  "ramp_time_s": 30.0,
  "hold_time_s": 60.0,
  "magnitude_mode": "percent_of_room_load",
  "load_percentage": 0.1,
  "reference_load_path": "boundary_conditions.load_before_w"
}
```

With `magnitude_mode: "percent_of_room_load"`, the disturbance magnitude is
resolved after startup initialization, so `load_percentage: 0.1` uses 10% of
the startup-solved `boundary_conditions.load_before_w`. The disturbance is then
applied as an equal-and-opposite room-to-dock exchange pulse: the room gets a
positive load and the dock gets the same magnitude with the opposite sign.
The resolved value is written to CSV as `infiltration_magnitude_w`.

Use `magnitude_mode: "fixed_w"` with `magnitude_w` to keep a fixed calibration
size instead.

The disturbance starts only after `start_time_s + delay_s`, ramps up linearly over
`ramp_time_s`, stays at full magnitude for `hold_time_s`, then ramps back down to
zero over the same `ramp_time_s`.

## PI Control

PI controllers are configured in JSON under `control.controllers`.
Each controller reads one output from the previous time step and writes one actuator path before the next Newton solve.

The implemented discrete form is:

```text
error = measurement - setpoint
if action == "reverse": error = -error
integral = integral + error * dt
target = bias + gain * error + gain * integral / Ti
target = clamp(target, u_min, u_max)
output = apply optional actuator lag and rate limit
output = clamp(output, u_min, u_max)
```

The current controller mapping in the reference case is:

- `B1_air_before_room_to_air_compressor_speed`: air temperature before entering the room (`t5_c`) to air-compressor speed
- `B2_air_after_cascade_to_nh3_compressor_speed`: air temperature after leaving the cascade exchanger (`t3_c`) to NH3 compressor speed
- `B3_superheat_to_expansion_valve`: refrigerant superheat to expansion-valve opening

So the air-cycle compressor speed regulates the air supplied to the room, the
NH3 compressor speed regulates the air temperature after the cascade exchanger,
and the expansion valve regulates refrigerant superheat.

## Notes

- The paper title used as design context is:
  `Cascade refrigeration system with inverse Brayton cycle on the cold side`
- `config/paper_reference_case.json` uses values recovered from Table 2 of the paper, then translates them into this simplified transient model.
- Because the paper is primarily a steady-state design study and includes a loading-dock evaporator branch that is represented here with a lumped dynamic model, a few values are derived approximations:
  - Brayton pressure ratio from the paper temperatures and turbine efficiency under the humid-air expansion model
  - ammonia compressor performance represented by a variable-speed BITZER map
  - UA values back-calculated from reference duties and terminal temperatures
- The code is intentionally modular so you can later add more detailed control volumes, pressure dynamics, frost, or more detailed heat exchanger discretization.
