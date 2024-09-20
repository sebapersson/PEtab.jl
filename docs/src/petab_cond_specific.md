# [Simulation Condition-Specific Model Parameters](@id define_conditions)

Sometimes, a subset of model parameters to be estimated can have different values across experimental conditions. For example, for a parameter estimation problem the model parameter `c1` might have one value for condition `cond1` and a different value for condition `cond2`.

This tutorial covers how to handle condition-specific parameters to estimate when creating a `PEtabModel`. It requires that you are familiar with PEtab simulation conditions; if not, see this [tutorial](add). As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial]. Even though the code below encodes the model as a `ReactionSystem`, everything works exactly the same if the model is encoded as an `ODESystem`.

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

p_S0 = PEtabParameter(:S0)
p_c2 = PEtabParameter(:c2)
p_sigma = PEtabParameter(:sigma)

default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm) # hide
nothing # hide
```

## Specifying Condition-Specific Parameters

Condition-specific parameters are handled by first defining them as `PEtabParameter`, and then linking the model parameter to the appropriate `PEtabParameter` when defining the simulation conditions. For example, assume that for condition `cond1`, the value for the model parameter `c1` should be given by `c1_cond1`, whereas for condition `cond2`, it should be given by `c1_cond2`. The first step is to define `c1_cond1` and `c1_cond2` as `PEtabParameter`:

```@example 1
p_c1_cond1 = PEtabParameter(:c1_cond1)
p_c1_cond2 = PEtabParameter(:c1_cond2)
pest = [p_c1_cond1, p_c1_cond2, p_S0, p_c2, p_sigma]
```

Next, when defining the simulation conditions, the model parameter `c1` must be mapped to the correct `PEtabParameter`:

```@example 1
cond1 = Dict(:E => 5.0, :c1 => :c1_cond1)
cond2 = Dict(:E => 2.0, :c1 => :c1_cond2)
conds = Dict("cond1" => cond1, "cond2" => cond2)
```

Note that each simulation condition also defines the initial value for specie `E`. Finally, as usual, each measurement must be assigned to a simulation condition:

```@example 1; ansicolor=false
using DataFrames
measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                         obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
```

Given a `Dict` with simulation conditions and `measurements` in the correct format, it is straightforward to create a PEtab problem with condition-specific parameters by simply providing the condition `Dict` under the `simulation_conditions` keyword:

```@example 1; ansicolor=false
model = PEtabModel(sys, observables, measurements, pest;
                   simulation_conditions = conds)
petab_prob = PEtabODEProblem(model)
```

With this setup, when simulating the model for `cond1`, the value of the model parameter `c1` is given by `c1_cond1`, and for `cond2`, it is given by `c1_cond2`. Furthermore, `c1_cond1` and `c1_cond2` are estimated during parameter estimation.

When working with models that have many condition-specific parameters, performance may improve by setting `split_over_conditions=true` (PEtab.jl tries to determine when to do this automatically, but it is a hard problem) when building the `PEtabODEProblem`. For more information on this, see [this](@ref Beer_tut) example.

## Additional Possible Configurations

In this tutorial example, the condition-specific parameters `[c1_cond1, c1_cond2]` map to one model parameter. It is also possible for condition specific parameters to map to multiple parameters. For example, the following is allowed:

```julia
cond1 = Dict(:c1 => :c1_cond1, :c2 => :c1_cond1)
```
