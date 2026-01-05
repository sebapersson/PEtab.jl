```@meta
CollapsedDocStrings=true
```

# [Events/callbacks](@id define_events)

Discrete changes to the ODE simulation that update states and/or parameters (e.g. a dose, a
media change, or a temperature shift) can be modeled with **events** (often called callbacks
in the SciML ecosystem). In PEtab.jl, events are specified with `PEtabEvent`:

```@docs; canonical=false
PEtabEvent
```

This tutorial shows how to specify both time-triggered events and state-triggered events
(e.g. when a state crosses a threshold), as well as how to assign condition specific events.
As a running example, we use the Michaelis–Menten model from the [starting tutorial](@ref
tutorial).

```@example 1
using Catalyst, PEtab

rn = @reaction_network begin
    @parameters begin
      S0
      c3 = 1.0
    end
    @species begin
      S(t) = S0
      E(t) = 50.0
      P(t) = 0.0
    end
    @observables begin
      obs1 ~ S + E
      obs2 ~ P
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

@parameters sigma
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
observables = [petab_obs1, petab_obs2]

# Value are set for better plots
p_c1 = PEtabParameter(:c1; value = 1.0)
p_c2 = PEtabParameter(:c2; value = 2.0)
p_s0 = PEtabParameter(:S0; value = 5.0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

measurements = DataFrame(obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
using Plots # hide
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
nothing # hide
```

!!! note "Why `PEtabEvent`?" While events/callbacks can be directly encoded in a Catalyst
`ReactionNetwork` or a `ODESystem` model, we strongly recommend using `PEtabEvent` for
optimal performance, and to ensure the correct evaluation of the objective function and in
particular its gradient especially its [frohlich2017parameter](@cite).

## Time-triggered events

Time-triggered events fire at a specified trigger time. The trigger time can be a constant
(e.g. `t == 2.0`) or a model parameter (e.g. `t == c2`). For example, to trigger an event at
`t = 2.0` that sets the state `S` to `2.0`:

```@example 1
t = default_t()
@unpack S = rn
event = PEtabEvent(t == 2.0, :S => S + 2)
```

A `PEtabModel` accounting for this event can be created by passing the event via the
`events` keyword:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events = event)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

From simulating the model, we see the jump in `S` at `t = 2.0`:

```@example 1
using Plots
x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

The trigger time can also be a model parameter (which can be estimated). For example, to
trigger when `t == c2` and set `c1 => 2.0`:

```@example 1
@unpack c2 = rn
event = PEtabEvent(t == c2, :c1 => 2.0)
model = PEtabModel(rn, observables, measurements, pest; events = event)
petab_prob = PEtabODEProblem(model)
x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

## State-triggered events

State-triggered events fire when the event condition transitions from `false` to `true`. For
example, to set `S => 1.0` when the state `S` reaches `0.2`:

```@example 1
@unpack S = rn
event = PEtabEvent(S ≥ 0.2, :S => 1.0)
```

Plotting the solution, we see the jump when `S = 0.2`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events = event)
petab_prob = PEtabODEProblem(model)
x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

Because events trigger on a `false → true` transition, event directionality can be
controlled using inequalities. For example, if `S` starts above `0.2`, then `S > 0.2` is
already `true` at `t0` and the event will not trigger:

```@example 1
event = PEtabEvent(S > 0.2, :S => 1.0)
model = PEtabModel(rn, observables, measurements, pest; events = event)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

Meanwhile, with condition `S < 0.2` the event triggers when `S` goes below `0.2`:

```@example 1
event = PEtabEvent(S < 0.2, :S => S + 1)
model = PEtabModel(rn, observables, measurements, pest; events = event)
petab_prob = PEtabODEProblem(model)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

## Multiple event targets

If an event updates multiple states and/or parameters, provide multiple assignments. For
example, to set both `S => S + 2` and `c1 => 2.0` when `S < 0.2`:

```@example 1
@unpack S = rn
event = PEtabEvent(S < 0.2, :S => S + 2, :c1 => 2.0)
```

The event is then provided as usual to `PEtabModel`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events = event)
petab_prob = PEtabODEProblem(model)
x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

## Multiple events

If a model has multiple events, pass them as a `Vector{PEtabEvent}`. For example, let
`event1` trigger when `S < 0.2` setting `S => S + 2`, and let `event2` trigger at `t = 1.0`
setting `c1 => 2.0`:

```@example 1
t = default_t()
@unpack S = rn
event1 = PEtabEvent(S < 0.2, :S => S + 2)
event2 = PEtabEvent(t == 1.0, :c1 => 2.0)
events = [event1, event2]
nothing # hide
```

The, the events can be provided as usual via the `events` keyword to `PEtabModel`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events = events)
petab_prob = PEtabODEProblem(model)
x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
plot(sol; linewidth = 2.0)
```

## Simulation condition specific events

An event can be made specific to a simulation condition via the `condition_ids` keyword. For
example, assume two simulation conditions:

```@example 1
cond1 = PEtabCondition(:cond1, :S => 5.0)
cond2 = PEtabCondition(:cond2, :S => 2.0)
simulation_conditions = [cond1, cond2]
measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                         obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
nothing # hide
```

And that the event is only triggered in the second condition:

```@example 1
event = PEtabEvent(t == 2.0, :S => 3.0; condition_ids = [:cond2])
```

From plotting the solution, the event clearly only triggers in `:cond2`:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; events=event
    simulation_conditions = simulation_conditions)
petab_prob = PEtabODEProblem(model)

sol_cond1 = get_odesol(x, petab_prob; condition = :cond1)
sol_cond2 = get_odesol(x, petab_prob; condition = :cond2)
p1 = plot(sol_cond1, title = "cond1")
p2 = plot(sol_cond2, title = "cond2")
plot(p1, p2)
plot(p1, p2; size = (800, 400)) # hide
```

## References

```@bibliography
Pages = ["petab_event.md"]
Canonical = false
```
