# [Events (callbacks, dosages, etc.)](@id define_events)

To account for experimental interventions, such as the addition of a substrate, changes in experimental conditions (e.g., temperature), or automatic dosages when a specie exceeds a certain concentration, events (often called [callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/), dosages, etc.) can be used. When creating a `PEtabModel` in Julia, events should be encoded as a `PEtabEvent`:

```@docs
PEtabEvent
```

This tutorial covers how to specify two types of events: those triggered at specific time points and those triggered by a species (e.g., when a species exceeds a certain concentration). As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial]. Even though the code below provides the model as a `ReactionSystem`, everything works exactly the same if the model is provided as an `ODESystem`.

```@example 1
using Catalyst, PEtab, Plots

t = default_t()
rn = @reaction_network begin
    @parameters begin
        S0
        c3 = 1.0
    end
    @species begin
        SE(t) = S0
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

@unpack E, S = rn
@parameters sigma
obs_sum = PEtabObservable(S + E, 3.0)
obs_p = PEtabObservable(P, sigma)
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)

p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

# Slimmed down compared to tutorial
measurements = DataFrame(
    obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5]
)
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm) # hide
nothing # hide
```

!!! note
    Events/callbacks can be directly encoded in a Catalyst `ReactionNetwork` or a ModelingToolkit `ODESystem` model. However, we strongly recommend using `PEtabEvent` for optimal performance and to ensure the correct evaluation of the objective function and its derivatives.

## Time-Triggered Events

Time-triggered events are activated at specific time points. The trigger value can be either a constant value (e.g., `t == 2.0`) or a model parameter (e.g., `t == c2`). For example, to trigger an event at `t = 2`, where species `S` is updated as `S <- S + 2`, do:

```@example 1
@unpack S = rn
event = PEtabEvent(t == 2.0, S + 2, S)
```

Then, to ensure the event is included when building the `PEtabModel`, include the `event` under the `events` keyword:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

From solving the dynamic ODE model with the parameter vector `x = [c2=0.1, c3=0.1, se0=0.1]`, it is clear that at `t == 2`, `S` is incremented by 2:

```@example 1
p = [0.1, 0.1, 0.1]
sol = get_sol(p, petab_prob)
plot(sol)
```

The trigger time can also be a model parameter, where the parameter is allowed to be estimated. For instance, to trigger the event when `t == c2` and assign `c1` as `c1 <- 2.0`, do:

```@example 1
@unpack c2 = rn
event = PEtabEvent(t == c2, 2.0, :c1)
```

From plotting the solution, it is clear that a change in dynamics occurs at `t == c2`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event)
petab_prob = PEtabODEProblem(model)
sol = get_sol(p, petab_prob)
plot(sol)
```

!!! note
    If the condition and target are single parameters or species, they can be specified as `Num` (from `@unpack`) or a `Symbol` (`:c1` above). If the event involves multiple parameters or species, they must be provided as a `Num` equation (see below).

## Specie-Triggered Events

Specie-triggered events are activated when a species-dependent Boolean condition transitions from `false` to `true`. For example, suppose we have a dosage machine that triggers when the substrate `S` drops below the threshold value of `0.2`, and at that point, the machine updates `S` as `S <- S + 1`. This can be encoded as:

```@example 1
@unpack S = rn
event = PEtabEvent(S == 0.2, 1.0, S)
```

Plotting the solution, we can clearly see how `S` is incremented every time it reaches `0.2`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event)
petab_prob = PEtabODEProblem(model)
sol = get_sol(p, petab_prob)
plot(sol)
```

With species-triggered events, the direction can matter. For instance, with `S == 0.2`, the event is triggered when `S` approaches `0.2` from either above or below. To activate the event only when `S` drops below `0.2`, write:

```@example 1
event = PEtabEvent(S < 0.2, 1.0, S)
model = PEtabModel(rn, observables, measurements, pest; events=event)
petab_prob = PEtabODEProblem(model)
sol = get_sol(p, petab_prob)
plot(sol)
```

Because events only trigger when the condition (`S < 0.2`) transitions from `false` to `true`, this event is triggered when `S` approaches from above. Meanwhile, if we write `S > 0.2`, the event is never triggered:

```@example 1
event = PEtabEvent(S > 0.2, 1.0, S)
model = PEtabModel(rn, observables, measurements, pest; events=event)
petab_prob = PEtabODEProblem(model)
sol = get_sol(p, petab_prob)
plot(sol)
```

## Multiple Event Targets

Sometimes an event can affect multiple species and/or parameters. In this case, both `affect` and `target` should be provided as vectors. For example, suppose an event is triggered when the substrate fulfills`S < 0.2`, where `S` is updated as `S <- S + 2` and `c1` is updated as `c1 <- 2.0`. This can be encoded as:

```@example 1
event = PEtabEvent(S < 0.2, [S + 2, 2.0], [S, c1])
```

The event is provided as usual to the `PEtabModel`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event)
petab_prob = PEtabODEProblem(model)
sol = get_sol(p, petab_prob)
plot(sol)
```

When there are multiple targets, the length of the `affect` vector must match the length of the `target` vector.

## Multiple Events

Sometimes a model can have multiple events, which then should be provided as `Vector` of `PEtabEvent`. For example, suppose `event1` is triggered when the substrate fulfills `S < 0.2`, where `S` is updated as `S <- S + 2`, and `event2` is triggered when `t == 1.0`, where `c1` is updated as `c1 <- 2.0`. This can be encoded as:

```@example 1
@unpack S, c1 = rn
event1 = PEtabEvent(S < 0.2, 1.0, S)
event2 = PEtabEvent(1.0, 2.0, :c1)
events = [event1, event2]
```

These events are then provided as usual to the `PEtabModel`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=events)
petab_prob = PEtabODEProblem(model)
sol = get_sol(p, petab_prob)
plot(sol)
```

In this example, two events are provided, but with your imagination as the limit, any number of events can be provided.

## Modifying Event Parameters for Different Simulation Conditions

The trigger time (`condition`) and/or `affect` can be made specific to different simulation conditions by introducing control parameters (here `c_time` and `c_value`) and setting their values accordingly in the simulation conditions:

```@example 1
rn = @reaction_network begin
    @parameters se0 c_time c_value
    @species SE(t) = se0  # se0 represents the initial value of S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Define state and parameter maps
statemap =  [:E => 1.0, :P => 0.0]
# c_value defaults to 1 prior to the event trigger
parametermap = [:c1 => 1.0]

condition_c0 = Dict(:S => 5.0, :c_time => 1.0, :c_value => 2.0)
condition_c1 = Dict(:S => 2.0, :c_time => 4.0, :c_value => 3.0)
simulation_conditions = Dict("cond0" => condition_c0,
                             "cond1" => condition_c1)
```

In this setup, when the event is defined as:

```@example 1
event = PEtabEvent(:c_time, :c_value, :c1)
```

the `c_time` parameter controls when the event is triggered, so for condition `c0`, the event is triggered at `t=1.0`, while for condition `c1`, it is triggered at `t=4.0`. Additionally, for conditions `cond0` and `cond1`, the parameter `c1` takes on the corresponding `c_value` values, which is `2.0` and `3.0`, respectively, which can clearly be seen when plotting the solution for `cond0`

```@example 1
model = PEtabModel(
    rn, simulation_conditions, observables, measurements,
    parameters, statemap=statemap, parametermap=parametermap,
    events=event, verbose=false
)
petab_prob = PEtabODEProblem(model, verbose=false)
sol = get_sol(p, petab_prob; condition_id=:cond0)
plot(sol)
```

and `cond1`

```@example 1
sol = get_sol(p, petab_prob; condition_id=:cond1)
plot(sol)
```
