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
  - condenser sink outlet / lumped sink temperature
- Algebraic variables:
  - refrigerated room / compressor inlet temperature from the room heater load
  - air-cycle regenerator outlet temperatures
  - air temperature entering the turbine
  - air supply temperature to the cold room
  - ammonia evaporating temperature
  - ammonia mass flow through the expansion valve

Component models:

- compressor / turbine: isentropic efficiency
- regenerator / cascade HX / condenser: `UA-LMTD`
- room load: heater model, `Q_RS = m_air * (h_room_return - h_supply)`
- loading dock branch: direct heater/load on the refrigerant side, `Q_LD(t)`
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

The bottom-side air mass flow is computed from the compressor isentropic head:

```text
h_is = h(P2, s1) - h1
H = h_is / g
Q = f(H, rpm)
m_air = rho_suction * Q
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
separate startup design-point solve. This mode does not advance time. It fixes
the configured targets, solves selected startup-only free variables, writes those
values back into the in-memory config, then freezes them for the dynamic time
loop. If `freeze_solved_parameters` is true, controllers whose actuator path was
solved during startup report the frozen value instead of overwriting it.

The startup problem is configured by:

- `targets`: design-point temperatures for room, sink, evaporator, and
  condenser
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

```json
{
  "enabled": true,
  "start_time_s": 900.0,
  "delay_s": 120.0,
  "ramp_time_s": 30.0,
  "hold_time_s": 60.0,
  "magnitude_w": 10000.0,
  "room_fraction": 1.0,
  "dock_fraction": -1.0
}
```

The disturbance starts only after `start_time_s + delay_s`, ramps up linearly over
`ramp_time_s`, stays at full magnitude for `hold_time_s`, then ramps back down to
zero over the same `ramp_time_s`. With the fractions above, it increases room load
and decreases dock load by the same amount.

## PI Control

PI controllers are configured in JSON under `control.controllers`.
Each controller reads one output from the previous time step and writes one actuator path before the next Newton solve.

The implemented discrete form is:

```text
error = measurement - setpoint
if action == "reverse": error = -error
integral = integral + error * dt
output = bias + gain * error + gain * integral / Ti
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
- Because the paper is primarily a steady-state design study and includes humid-air expansion details and a loading-dock evaporator branch that are not yet fully represented here, a few values are derived approximations:
  - Brayton pressure ratio from the paper temperatures and turbine efficiency under a dry-air fit
  - ammonia compressor performance represented by a variable-speed BITZER map
  - UA values back-calculated from reference duties and terminal temperatures
- The code is intentionally modular so you can later add more detailed control volumes, pressure dynamics, frost, or more detailed heat exchanger discretization.
