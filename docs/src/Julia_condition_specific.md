# [Condition Specific System/Model Parameters](@id define_conditions)

In some cases, certain model such like a substrate enzyme binding rate may vary between experimental conditions while other parameters remain constant. These condition-specific parameters can be defined via the simulation conditions. To demonstrate how, let us consider the same enzyme kinetics model as used in the [Creating a PEtab Parameter Estimation Problem in Julia](@ref define_in_julia) tutorial.

```julia
using Catalyst
using DataFrames
using Distributions
using PEtab

rn = @reaction_network begin
    @parameters se0
    @species SE(t) = se0  # se0 = initial value for S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Define state and parameter maps
state_map = [:E => 1.0, :P => 0.0]
parameter_map = [:c2 => 1.0]

# Unpack model components
@unpack P, E, SE = rn
@parameters sigma, scale, offset

# Define observables
obs_P = PEtabObservable(scale * P + offset, sigma * P, transformation=:lin)
obs_Sum = PEtabObservable(E + SE, 3.0, transformation=:log)
observables = Dict("obs_P" => obs_P,
                   "obs_Sum" => obs_Sum)

measurements = DataFrame(
    simulation_id=["cond0", "cond0", "cond1", "cond1"],
    obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5]
)
```

Now, assume that for condition `cond0`, the substrate binding rate `c1` should have a different value than under simulation condition `cond1`. This can be defined as follows:

```julia
_c3 = PEtabParameter(:c3, scale=:log10)
_se0 = PEtabParameter(:c3, prior=LogNormal(1.0, 0.5), prior_on_linear_scale=true)
_sigma = PEtabParameter(:sigma)
_scale = PEtabParameter(:scale)
_offset = PEtabParameter(:offset)
_c1_cond0 = PEtabParameter(:c1_cond0)
_c1_cond1 = PEtabParameter(:c1_cond1)
parameters = [_c1_cond0, _c1_cond1, _se0, _sigma, _scale, _offset]

# Define simulation conditions
condition_c0 = Dict(:S => 5.0, :c1 => :c1_cond0)
condition_c1 = Dict(:S => 2.0, :c1 => :c1_cond1)
simulation_conditions = Dict("cond0" => condition_c0,
                             "cond1" => condition_c1)
```

When simulating the model, the value of `c1_cond0` is used for simulation condition `cond0`, and the value of `c1_cond1` is used for simulation condition `cond1`. With this setup, you can create a `PEtabODEProblem` for model calibration:

```julia
petab_model = PEtabModel(
    rn, simulation_conditions, observables, measurements,
    parameters, state_map=state_map, parameter_map=parameter_map, verbose=true
)
petab_problem = PEtabODEProblem(petab_model)
```

Note that for models with many conditions specific parameters performance can be improved by setting `split_over_conditions=true` when building the `PEtabODEProblem`, for additional information see [this](@ref Beer_tut) tutorial.