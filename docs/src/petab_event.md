```@meta
CollapsedDocStrings=true
```

# [Events (callbacks, dosages, etc.)](@id define_events)

To account for experimental interventions, such as the addition of a substrate, changes in experimental conditions (e.g., temperature), or automatic dosages, events (often called [callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/), dosages, etc.) can be used. When creating a `PEtabModel` in Julia, events should be encoded as a `PEtabEvent`:

```@docs; canonical=false
PEtabEvent
```

This tutorial covers how to specify two types of events: those triggered at specific time points and those triggered by a species (e.g., when a specie exceeds a certain concentration). As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial](@ref tutorial). Even though the code below provides the model as a `ReactionSystem`, everything works exactly the same if the model is provided as an `ODESystem`.

```@example 1
using Catalyst, DataFrames, PEtab

rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 2.0, :SE => 0.0, :P => 0.0]

@unpack E, S, P = rn
@parameters sigma
obs_sum = PEtabObservable(S + E, 3.0)
obs_p = PEtabObservable(P, sigma)
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)

# Set values for better plots
p_c1 = PEtabParameter(:c1; value = 1.0)
p_c2 = PEtabParameter(:c2; value = 2.0)
p_s0 = PEtabParameter(:S0; value = 5.0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

# Smaller dataset compared to starting tutorial
measurements = DataFrame(obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
using Plots # hide
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
nothing # hide
```

!!! note
    Events/callbacks can be directly encoded in a Catalyst `ReactionNetwork` or a ModelingToolkit `ODESystem` model. However, we strongly recommend using `PEtabEvent` for optimal performance, and to ensure the correct evaluation of the objective function and especially its derivative [frohlich2017parameter](@cite).

## Time-Triggered Events

Time-triggered events are activated at specific time points. The trigger value can be either a constant value (e.g., `t == 2.0`) or a model parameter (e.g., `t == c2`). For example, to trigger an event at `t = 2`, where species `S` is updated as `S <- S + 2`, do:

```@example 1
@unpack S = rn
event = PEtabEvent(t == 2.0, S + 2, S)
```

Then, to ensure the event is included when building the `PEtabModel`, include the `event` under the `events` keyword:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

From solving the dynamic ODE model, it is clear that at `t == 2`, `S` is incremented by 2:

```@example 1
using Plots
x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

The trigger time can also be a model parameter, where the parameter is allowed to be estimated. For instance, to trigger the event when `t == c2` and assign `c1` as `c1 <- 2.0`, do:

```@example 1
@unpack c2 = rn
event = PEtabEvent(t == c2, 5.0, :c1)
```

From plotting the solution, it is clear that a change in dynamics occurs at `t == c2`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

!!! note
    If the condition and target are single parameters or species, they can be specified as `Num` (from `@unpack`) or a `Symbol` (`:c1` above). If the event involves multiple parameters or species, they must be provided as a `Num` equation (see below).

## Specie-Triggered Events

Specie-triggered events are activated when a species-dependent Boolean condition transitions from `false` to `true`. For example, suppose we have a dosage machine that triggers when the substrate `S` drops below the threshold value of `2.0`, and at that point, the machine updates `S` as `S <- S + 1`. This can be encoded as:

```@example 1
@unpack S = rn
event = PEtabEvent(S == 0.2, 1.0, S)
```

Plotting the solution, we can clearly see how `S` is incremented every time it reaches `0.2`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

With species-triggered events, the direction can matter. For instance, with `S == 0.2`, the event is triggered when `S` approaches `0.2` from either above or below. To activate the event only when `S` drops below `0.2`, write:

```@example 1
event = PEtabEvent(S < 0.2, 1.0, S)
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

Because events only trigger when the condition (`S < 0.2`) transitions from `false` to `true`, this event is triggered when `S` approaches from above. Meanwhile, if we write `S > 0.2`, the event is never triggered:

```@example 1
event = PEtabEvent(S > 0.2, 1.0, S)
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

## Multiple Event Targets

Sometimes an event can affect multiple species and/or parameters. In this case, both `affect` and `target` should be provided as vectors. For example, suppose an event is triggered when the substrate fulfills`S < 0.2`, where `S` is updated as `S <- S + 2` and `c1` is updated as `c1 <- 2.0`. This can be encoded as:

```@example 1
event = PEtabEvent(S < 0.2, [S + 2, 2.0], [S, :c1])
```

The event is provided as usual to the `PEtabModel`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
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
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

In this example, two events are provided, but with your imagination as the limit, any number of events can be provided.

## Modifying Event Parameters for Different Simulation Conditions

The trigger time (`condition`) and/or `affect` can be made specific to different simulation conditions by introducing control parameters (here `c_time` and `c_value`) and setting their values accordingly in the simulation conditions:

```@example 1
rn = @reaction_network begin
    @parameters c3=1.0 S0 c_time c_value
    @species S(t) = S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

cond1 = Dict(:S => 5.0, :c_time => 1.0, :c_value => 2.0)
cond2 = Dict(:S => 2.0, :c_time => 4.0, :c_value => 3.0)
conds = Dict("cond1" => cond1, "cond2" => cond2)

measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                         obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
nothing # hide
```

In this setup, when the event is defined as:

```@example 1
event = PEtabEvent(:c_time, :c_value, :c1)
```

the `c_time` parameter controls when the event is triggered, so for condition `c0`, the event is triggered at `t=1.0`, while for condition `c1`, it is triggered at `t=4.0`. Additionally, for conditions `cond1` and `cond2`, the parameter `c1` takes on the corresponding `c_value` values, which is `2.0` and `3.0`, respectively, which can clearly be seen when plotting the solution for `cond1`

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event, speciemap = speciemap,
                   simulation_conditions = conds)
petab_prob = PEtabODEProblem(model)
sol_cond1 = get_odesol(x, petab_prob; cid=:cond1)
plot(sol_cond1; linewidth = 2.0)
```

and `cond2`

```@example 1
sol = get_odesol(x, petab_prob; cid=:cond2)
plot(sol; linewidth = 2.0)
```

## References

```@bibliography
Pages = ["petab_event.md"]
Canonical = false
```
