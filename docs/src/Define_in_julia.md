# [Creating a PEtab Parameter Estimation Problem in Julia](@id define_in_julia)

While the PEtab table format is excellent for specifying parameter estimation problems for dynamic ODE models, setting up a parameter estimation problem directly in Julia can be more convenient.

Here we demonstrate how to define a parameter estimation problem using a simple Micheli-Mentan model as an example. We will discuss in detail the five essential components required to define a problem:

1. **Dynamic Model**: You can use either a `ReactionSystem` defined in [Catalyst](https://petab.readthedocs.io/en/latest/) or an `ODESystem` defined in [ModellingToolkit](https://github.com/SciML/ModelingToolkit.jl).
2. **Observable Formula**: To link the model to the measurement data, you need an observable formula. Since real-world data often comes with measurement noise, you also must specify a noise formula and noise distribution. This is specified as a `PEtabObservable`.
3. **Parameters to Estimate**: Typically, you do not want to estimate all model parameters. Moreover, sometimes you might want to incorporate prior beliefs by assigning priors to certain parameters. Parameter information is provided as a vector of `PEtabParameter`.
4. **Simulation Conditions**: Measurements are often taken under various experimental conditions, such as different substrate concentrations. These experimental conditions typically correspond to model control parameters, like the initial value of a model species. You specify these conditions as a `Dict` (see below). In case the model only has a single simulation conditions, `simulation_conditions` can be omitted when building the `PEtabModel`.
5. **Measurement Data**: To calibrate the model, you need measurement data, which should be provided as a `DataFrame`. The data format is explained below.

## Defining the Dynamic Model

To define the dynamic model, you have two options; you can use a [Catalyst](https://petab.readthedocs.io/en/latest/) defined `ReactionSystem`, or a [ModellingToolkit](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`. Using Catalyst we define the model as

```julia
using Catalyst
using PEtab

rn = @reaction_network begin
    @parameters se0
    @species SE(t) = se0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
```

If you want to estimate the initial value of a species (like `SE`), you must define it as a parameter, as here with `SE(t) = se0`.

Using a ModellingToolkit `ODESystem` we define the model as:

```julia
using ModellingToolkit

@parameters c1, c2, c3, se0
@variables t S(t) SE(t) P(t) E(t)
D = Differential(t)
eqs = [
    D(S) ~ -c1*S*E + c2*SE,
    D(E) ~ -c1*S*E + c2*SE + c3*SE,
    D(SE) ~ c1*S*E - c2*SE - c3*SE,
    D(P) ~ c3*SE
]
@named sys = ODESystem(eqs; defaults=Dict(SE => se0))
```

To estimate an initial value, such as `SE`, for an `ODESystem`, you need to define it using a dictionary under the `defaults` keyword, here done via `defaults = Dict(SE => se0)`.

Regardless of how the model is defined, if you want to fixate a parameter or initial value to a constant value across all simulations, you can use a state and/or parameter map. For instance, to set `E` and `P` to be 1.0 and 0.0, and set `c1` to 1.0, do:

```julia
state_map = [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]
```

If a parameter or initial value is not specified anywhere it defaults to zero.

## Defining the Observable

To connect our model with measurement data, we need an observable formula. Since data from a reaction networks typically includes measurement noise, we also require a noise formula and a noise distribution.

Let us assume we are observing the product `P` with a normally distributed multiplicative measurement error (`sigma * P`) on a relative scale. To account for this relative scale we can as commonly done use `scale` and `offset` parameters. Additionally, let us assume we are directly measure the sum `E + SE` with log-normal measurement noise, and we already know the measurement error (`sigma`) is 3.0. This can be defined as

```julia
@unpack P, E, SE = rn
@parameters sigma, scale, offset
obs_P = PEtabObservable(scale * P + offset, sigma * P, transformation=:lin)
obs_Sum = PEtabObservable(E + SE, 3.0, transformation=:log)
```

The `transformation` parameter can take one of three values: `:lin` (default for normal measurement noise), `:log`, or `:log10` (for log-normal measurement noise).

To complete the definition, we group these observables together in a `Dict` with appropriate names:

```julia
observables = Dict("obs_P" => obs_P,
                   "obs_Sum" => obs_Sum)
```

## Defining Parameters to Estimate

To set up a parameter estimation problem, we need to specify the parameters to estimate. To improve the estimation it is often beneficial to define lower and upper bounds to restrict the parameter space. For example, let us assume we want to estimate the parameter `c3` with bounds `[1e-3, 1e3]` (default):

```julia
_c3 = PEtabParameter(:c3, lb=1e-3, ub=1e3, scale=:log10)
```

Here `scale=:log10` means that we are estimating the parameter on a log10 scale, which typically yields better results than a linear-scale. Overall the `scale` parameter can take on three values: `:lin`, `:log`, and `:log10` (default).

If you have prior information about parameters, you can specify a continuous prior distribution from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For instance, if you want to estimate `se0` (the initial value of species `SE`) and you know it should be around 3.1, you can set a prior as:

```julia
using Distributions
_se0 = PEtabParameter(:se0, prior=LogNormal(log(3.1), 0.5), prior_on_linear_scale=true)
```

In this case, `prior_on_linear_scale=true` (default) indicates that the prior is defined on the linear scale, not the default transformed log10 scale used for parameter estimation.

Apart from estimating parameters in the reaction system, you can also estimate parameters related to measurement noise or parameters used exclusively in the observable formula (e.g., `scale` and `offset` parameters see above) by defining them as a `PEtabParameter`:

```julia
# Using default bounds [1e-3, 1e3] and scale=:log10
_sigma = PEtabParameter(:sigma)
_scale = PEtabParameter(:scale)
_offset = PEtabParameter(:offset)
```

Once the parameters are defined they should be gathered into a vector

```julia
_c2 = PEtabParameter(:c2)
parameters = [_c2, _c3, _se0, _sigma, _scale, _offset]
```

## Defining Simulation Conditions

Data is often collected under various experimental settings, such as different initial concentrations of a substrate. These variations in experimental conditions correspond to different simulation conditions during model calibration. To effectively align your measurements with the data, you need to specify these simulation conditions using a dictionary.

Specifically, assume you have measured your data under two conditions: `c0` and `c1`, where each condition has different starting concentrations for the substrate `S`. This can be defined as:

```julia
condition_c0 = Dict(:S => 5.0)
condition_c1 = Dict(:S => 2.0)
```

Here, the key (in this case, `S`) can represent either a model species (as in this case) or a parameter. To complete the setup, gather all the simulation conditions in a dictionary, and assign each condition an appropriate name:

```julia
simulation_conditions = Dict("c0" => condition_c0,
                             "c1" => condition_c1)
```

!!! note
    If a parameter or species is specified for one simulation condition, it must be specified for all simulation conditions.

## Defining Measurement Data

The measurement data should be organized as a `DataFrame` in the following format (the column names matter, but not the order)

| simulation_id (str) | obs_id (str) | time (float) | measurement (float) |
|---------------------|--------------|--------------|---------------------|
| c0                  | obs_P        | 1.0          | 0.7                 |
| c0                  | obs_Sum      | 10.0         | 0.1                 |
| c1                  | obs_P        | 1.0          | 1.0                 |
| c1                  | obs_Sum      | 20.0         | 1.5                 |

For each measurement, you need to specify:

- `simulation_id`: Identifies the simulation condition it corresponds to.
- `obs_id`: Specifies the observable it corresponds to.
- `time`: Indicates the time point at which the data was collected.
- `measurement`: The actual measurement value.

For this case, the input would look like;

```julia
using DataFrames

measurements = DataFrame(
    simulation_id=["c0", "c0", "c1", "c1"],
    obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5])
```

## Bringing It All Together

After defining the model, observables, parameters to estimate, simulation conditions, and measurement data, you can easily create a `PEtabODEProblem` for your parameter estimation task using the `ReactionSystem`:

```julia
petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                         parameters, state_map=state_map, parameter_map=parameter_map,
                         verbose=true)
petab_problem = PEtabODEProblem(petab_model, verbose=true)
```

or the `ODESystem`:

```julia
petab_model = PEtabModel(sys, simulation_conditions, observables, measurements,
                         parameters, state_map=state_map, parameter_map=parameter_map,
                         verbose=true)
petab_problem = PEtabODEProblem(petab_model, verbose=true)
```

The `PEtabODEProblem` contains all the necessary information to work with most available optimizers (see [here](@ref import_petab_problem)). Alternatively, if you want to perform parameter estimation using a multi-start approach, you can use the `calibrate_model_multistart` function (see see [Parameter estimation](@ref parameter_estimation)).

!!! note
    If the model does not have multiple simulation conditions (e.g., data is collected under a single condition), you can omit the `simulation_conditions` argument when constructing the `PEtabModel` and the `simulation_id` columns from the measurement data. Simply use the following format: `PEtabModel(sys, observables, measurements, parameters, <keyword arguments>)`.

## Where to Go Next

This example has covered the fundamental aspects of setting up a parameter estimation problem directly Julia, but there are additional options:

- **Steady-State Initialization**: In some cases, you might require your model to be at a steady-state at time zero when starting to match the model against data. To learn how to set up pre-equilibration criteria, see [this](@ref define_with_ss) tutorial.

- **Time-Point Specific Parameters**: You might measure the same observable with different assays, leading to different observable parameters (e.g., scale and offset) and noise parameters for various time points. To handle time-point-specific measurement and noise parameters, see [this](@ref time_point_parameters) tutorial.

- **Condition Specific System/Model Parameters**: Sometimes a subset of model parameters, like protein synthesis rates, vary between simulation conditions, while other parameters remain constant across all conditions. To handle conditions specific parameters, see [this](@ref define_conditions) tutorial.

- **Events**: Sometimes a model incorporates events like substrate addition at specific time points, and/or parameter changes when a state/species reaches certain values. To manage these events/callbacks, see [this](@ref define_events) tutorial.

For guidance on choosing the best options for your specific PEtab problem, we recommend the [Choosing the Best Options for a PEtab Problem](@ref best_options) section and refer to the [Supported Gradient and Hessian Methods](@ref gradient_support) section for more information on available gradient and hessian methods.

## Runnable Example

Here is the complete code from the tutorial

```julia
using Catalyst
using DataFrames
using Distributions
using ModellingToolkit
using PEtab

# Define the reaction network
rn = @reaction_network begin
    @parameters se0
    @species SE(t) = se0  # se0 = initial value for S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

@parameters c1, c2, c3, se0
@variables t S(t) SE(t) P(t) E(t)
D = Differential(t)
eqs = [
    D(S) ~ -c1*S*E + c2*SE,
    D(E) ~ -c1*S*E + c2*SE + c3*SE,
    D(SE) ~ c1*S*E - c2*SE - c3*SE,
    D(P) ~ c3*SE
]
@named sys = ODESystem(eqs; defaults=Dict(SE => se0))

# Define state and parameter maps
state_map =  [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]

# Unpack model components
@unpack P, E, SE = rn
@parameters sigma, scale, offset

# Define observables
obs_P = PEtabObservable(scale * P + offset, sigma * P, transformation=:lin)
obs_Sum = PEtabObservable(E + SE, 3.0, transformation=:log)
observables = Dict("obs_P" => obs_P,
                   "obs_Sum" => obs_Sum)

# Define parameters for estimation
_c3 = PEtabParameter(:c3, scale=:log10)
_se0 = PEtabParameter(:c3, prior=LogNormal(1.0, 0.5), prior_on_linear_scale=true)
_c2 = PEtabParameter(:c2)
_sigma = PEtabParameter(:sigma)
_scale = PEtabParameter(:scale)
_offset = PEtabParameter(:offset)
parameters = [_c2, _c3, _se0, _sigma, _scale, _offset]

# Define simulation conditions
condition_c0 = Dict(:S => 5.0)
condition_c1 = Dict(:S => 2.0)
simulation_conditions = Dict("c0" => condition_c0,
                             "c1" => condition_c1)

# Define measurement data
measurements = DataFrame(
    simulation_id=["c0", "c0", "c1", "c1"],
    obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5]
)

# Create a PEtab model. To build the petab_model with ODE-system instead of
# ReactionSystem provide sys instead of rn as first argument
petab_model = PEtabModel(
    rn, simulation_conditions, observables, measurements,
    parameters, state_map=state_map, parameter_map=parameter_map, verbose=true
)

# Create a PEtabODEProblem
petab_problem = PEtabODEProblem(petab_model)
```
