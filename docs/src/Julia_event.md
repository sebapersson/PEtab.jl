# [Events (callbacks, dosages etc...)](@id define_events)

To account for experimental interventions, such as the addition of a substrate, changes in experimental conditions (e.g., temperature), or automatic dosages when a species exceeds a certain concentration, events (often called callbacks, dosages etc...) can be used. In a `PEtabModel`, events can be encoded via `PEtabEvent`:

```julia
PEtabEvent(condition, affect, target)
```

where

- `condition` is the event trigger and can be i) a constant value or a model parameter that are assumed to be the time the event is activated, or ii) a Boolean expression triggered when, for example, a state exceeds a certain value.

- `affect` is the effect of the event, which can either be a value or an algebraic expression involving model parameters and/or states.

- `target` specifies the target on which the effect acts. It must be either a model state or parameter.

This section provides examples of how to use `PEtabEvent` to encode different types of events, and uses a modified version of the example in the [Creating a PEtab Parameter Estimation Problem in Julia](@ref define_in_julia) tutorial:

```julia
using Catalyst
using DataFrames
using Distributions
using PEtab


system = @reaction_network begin
    @parameters se0
    @species SE(t) = se0  # se0 = initial value for S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Define state and parameter maps
state_map =  [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]

# Define observables
@unpack P, E, SE = system
obs_P = PEtabObservable(P, 1.0, transformation=:lin)
obs_Sum = PEtabObservable(E + SE, 3.0, transformation=:log)
observables = Dict("obs_P" => obs_P,
                   "obs_Sum" => obs_Sum)

# Define parameters for estimation
_c3 = PEtabParameter(:c3, scale=:log10)
_se0 = PEtabParameter(:se0, prior=LogNormal(1.0, 0.5), prior_on_linear_scale=true)
_c2 = PEtabParameter(:c2)
petab_parameters = [_c2, _c3, _se0]

# Define simulation conditions
condition_c0 = Dict(:S => 5.0)
condition_c1 = Dict(:S => 2.0)
simulation_conditions = Dict("cond0" => condition_c0,
                             "cond1" => condition_c1)

# Define measurement data
measurements = DataFrame(
    simulation_id=["cond0", "cond0", "cond1", "cond1"],
    obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5]
)
```

In this section we cover two types of events: those triggered at specific time-points and those triggered by a state (e.g., when a state exceeds a certain concentration).

!!! note
    Even though events can be directly encoded in a Catalyst or ModellingToolkit model, we recommend `PEtabEvent` for optimal performance (e.g. use `DiscreteCallback` when possible), and for ensuring correct evaluation of the objective function and its derivatives.

## Time-Triggered Events

Time-triggered events are activated at specific time-points. The condition can either a constant value (e.g. 1.0) or a model parameter (e.g. `c2`). For example, to trigger an event at `t = 2` where the value 2 is added to the state `S` write:

```julia
@unpack S = system
event = PEtabEvent(2.0, S + 2, S)
```

When building the `PEtabModel` the event is then provided via the `events` keyword:

```julia
petab_model = PEtabModel(
    system, simulation_conditions, observables, measurements,
    petab_parameters, state_map=state_map, parameter_map=parameter_map,
    events=event, verbose=true
)
petab_problem = PEtabODEProblem(petab_model)
```

The trigger time can also be a model parameter. For instance, to trigger the event when `t == c2` and to set `c1` to 2.0 write:

```julia
event = PEtabEvent(:c2, 2.0, :c1)
```

!!! note
    If the condition and target are single parameters or states, they can be specied as `Num` (from unpack) or a `Symbol`.If the event involves multiple parameters or states, you must provide them as either a `Num` (as shown below) or a `String`.

### Modifying Event Parameters for Different Simulation Conditions

The trigger trime (`condition`) and/or `affect` can be made specific to different simulation conditions by introducing control parameters (here `c_time` and `c_value`) and setting their values accordingly in the simulation conditions:

```julia
system = @reaction_network begin
    @parameters se0 c_time c_value
    @species SE(t) = se0  # se0 represents the initial value of S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Define state and parameter maps
state_map =  [:E => 1.0, :P => 0.0]
# c_value defaults to 1 prior to the event trigger
parameter_map = [:c1 => 1.0]

condition_c0 = Dict(:S => 5.0, :c_time => 1.0, :c_value => 2.0)
condition_c1 = Dict(:S => 2.0, :c_time => 4.0, :c_value => 3.0)
simulation_conditions = Dict("cond0" => condition_c0,
                             "cond1" => condition_c1)
```

In this setup, when the event is defined as:

```julia
event = PEtabEvent(:c_time, :c_value, :c1)
```

the `c_time` parameter controls when the event is triggered, so for condition `c0`, the event is triggered at `t=1.0`, while for condition `c1`, it is triggered at `t=4.0`. Additionally, for conditions `cond0` and `cond1`, the parameter `c1` takes on the corresponding `c_value` values, which is `2.0` and `3.0`, respectively.

## State Triggered Events

State-triggered events are activated when a species/state fulfills a certain condition. For example, suppose we have a dosage machine that trigger when the substrate `S` drops below the threshold value of `0.1`, and at this time the machine adds up the substrate so it reaches the value 1.0:

```julia
@unpack S = system
event = PEtabEvent(S == 0.2, 1.0, S)
```

Here, `S == 0.1` means that the event is triggered when `S` reaches the value of 0.1.

With state-triggered events, the direction of the condition can matter. For instance, with `S == 0.2`, the event is triggered whether `S` approaches `0.2` from above or below. If we want it to activate only when `S` enters from above, write:

```julia
@unpack S = system
event = PEtabEvent(S < 0.2, 1.0, S)
```

Here, `S < 0.2` means that the event will trigger only when the expression goes from `false` to `true`, in this case when `S` approaches `0.2` from above. To trigger the event only when `S` approaches `0.2` from below, write `S > 0.2`.

### Modifying Event Parameters for Different Simulation Conditions

For models with multiple simulation conditions, it can be relevant to vary the trigger value and potentially change `affect` value for different simulations. This can be done via by introducing control parameters (here `s_trigger` and `s_vakye`) and setting their values accordingly in the simulation conditions:

```julia
system = @reaction_network begin
    @parameters se0 s_trigger s_value
    @species SE(t) = se0  # se0 = initial value for S
    c1 * c_controll, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Define state and parameter maps
state_map =  [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]

condition_c0 = Dict(:S => 5.0, :s_trigger => 0.2, :s_value => 1.0)
condition_c1 = Dict(:S => 2.0, :s_trigger => 0.3, :s_value => 2.0)
simulation_conditions = Dict("cond0" => condition_c0,
                             "cond1" => condition_c1)

@unpack S, s_trigger, s_value = system
event = PEtabEvent(S == s_trigger, s_value, S)
```

Now, for conditions `c0` and `c1`, the event is triggered when `S == 0.2` and `S == 0.3`, respectively. Additionally, the value of `S` changes to 1.0 and 2.0 for conditions `c0` and `c1`, respectively.

## Multiple Events

A model can have multiple events, which can be easily defined as a `PEtabModel` accepts a `Vector` of `PEtabEvent` as input. For example, suppose we have an event triggered when the substrate `S` satisfies `S < 0.2`, where `S` changes its value to `1.0`. Additionally, we have another event triggered when `t == 1.0`, where the parameter `c1` changes its value to `2.0`. This can be encoded as follows:

```julia
@unpack S, c1 = system
event1 = PEtabEvent(S < 0.2, 1.0, S)
event2 = PEtabEvent(1.0, 2.0, :c1)
events = [event1, event2]
```

These events can then be provided when building the `PEtabModel` with the `events` keyword:

```julia
petab_model = PEtabModel(
    system, simulation_conditions, observables, measurements,
    petab_parameters, state_map=state_map, parameter_map=parameter_map,
    events=events, verbose=true
)
petab_problem = PEtabODEProblem(petab_model)
```