# Breaking updates and feature summaries across releases

## PEtab.jl 4.0.0

PEtab.jl v4 is a breaking release driven by the PEtab standard format updating to format
v2. This release adds support for PEtab v2 while maintaining backward compatibility with
v1, and introduces a major usability-focused API cleanup. Major changes:

- Documentation overhaul (including migration to DocumenterVitepress).
- Internal refactor to improve maintainability; code relying on PEtab.jl internals
  may break.
- Unified Julia API for defining `PEtabModel`: `PEtabObservable`, `PEtabParameter`,
  `PEtabCondition` (simulation conditions), and `PEtabEvent` now follow consistent
  construction patterns.
- Removed the PyCall extension. Python-dependent functionality (PEtabSelect and Fides
  optimizers) now uses Julia wrapper packages (`PEtabSelect.jl`, `Fides.jl`) via
  PythonCall, removing the need to manage a Python environment manually.
- `remake` updated to support subsetting simulation conditions.
- Updated how simulation conditions are selected in `get_` and `plot` functions.
- Updated `plot`-recipes to support residuals of model fit.
- Improved printing of PEtab structs; added `describe` for `PEtabODEProblem`.

### Defining a `PEtabModel`

The API for specifying observables, parameters, conditions, and events has been unified.
Previously the following syntax was used to provide them to a `PEtabModel`:

- observables: `Dict{String,PEtabObservable}`
- parameters: `Vector{PEtabParameter}`
- conditions: `Dict{String,Dict}`
- events: `Vector{PEtabEvent}`

All of these are now provided as `Vector`s of the corresponding structs:
`Vector{PEtabObservable}`, `Vector{PEtabParameter}`, `Vector{PEtabCondition}`, and
`Vector{PEtabEvent}`. Constructors also follow a unified pattern,
`PEtab...(id, formulas...)`.

#### Observables

Observables are now defined as:

```julia
PEtabObservable(observable_id, observable_formula, noise_formula) # new
PEtabObservable(observable_formula, noise_formula)                # old
```

`PEtabObservable` no longer supports the `transformation` keyword. Instead, `distribution`
specifies the measurement noise model. For example, log-normal noise is defined as:

```julia
using Distributions
PEtabObservable(...; distribution = LogNormal) # new
PEtabObservable(...; transform = :log)         # old
```

Additional distributions are also supported (including `Laplace` and `LogLaplace`). Lastly,
observable formulas now support taking the ID to an observable defined in the model-system
(`ODESystem` or `ReactionSystem`) as input (see starting tutorial).

#### Parameters

Parameters are still defined with `PEtabParameter`. The keyword `prior_on_linear_scale` has
been removed: priors are now always interpreted on the linear scale. For example, if `c1`
is estimated on `log10` scale, the prior is placed on `c1` (not on `log(c1)`).

#### Simulation conditions

Simulation conditions are now defined with `PEtabCondition`:

```julia
cond1 = PEtabCondition(:cond1, target_id1 => target_value1, ...) # new
cond1 = Dict("cond1" => Dict(:target_id1 => target_value1, ...)) # old
```

Conditions are passed to `PEtabModel` as a `Vector{PEtabCondition}`. A non-zero simulation
start time can be set via `t0`, e.g. `PEtabCondition(...; t0=1.0)`. Formulas are also now
allowed in condition assignments:

```julia
@parameters k1 k2
cond1 = PEtabCondition(:cond1, :c1 => k1 + sin(k2))
```

#### Events

Events are defined as before, but the assignment syntax is now consistent with
`PEtabCondition`:

```julia
PEtabEvent(condition, target_id => target_value) # new
PEtabEvent(condition, target_value, target_id)   # old
```

Events can now be restricted to a subset of simulation conditions via `conditions`, e.g.
`PEtabEvent(...; conditions=[:cond1, :cond2])`.

### Selecting conditions in `get_` and `plot`

Condition selection keywords were updated to avoid ambiguous argument combinations. Without
pre-equilibration:

```julia
get_u0(x, prob; condition = :cond1) # new
get_u0(x, prob; cid = :cond1)       # old
```

With pre-equilibration:

```julia
get_u0(x, prob; condition = :pre_eq1 => :cond1) # new
get_u0(x, prob; cid = :cond1, pre_eq_id = :pre_eq1) # old
```

Plotting uses the same updated keyword:

```julia
plot(res, prob; condition = :cond1) # new
plot(res, prob; cid = :cond1)       # old
```

### `remake`

`remake` now supports subsetting both parameters and simulation conditions. Fixing
parameter values:

```julia
remake(prob; parameters = [:k1 => 3.0, :k2 => 4.0]) # new
remake(prob; xchange = Dict(:k1 => 1.0))            # old
```

Subsetting simulation conditions:

```julia
remake(prob; conditions = [:cond1, :cond2])  # without pre-equilibration
remake(prob; conditions = [:pre_eq1 => :cond1, :pre_eq2 => :cond2]) # with pre-equilibration
```

### Parameter estimation (Fides, PEtabSelect)

The trust-region Fides optimizers is now provided via
[Fides.jl](https://github.com/fides-dev/Fides.jl), removing the PyCall dependency:

```julia
# new
using Fides
res = calibrate(petab_prob, x0, Fides.BFGS())
# old
using PyCall
ENV["PYTHON"] = "path_to_python"
import Pkg; Pkg.build("PyCall")
res = calibrate(petab_prob, x0, Fides(:BFGS))
```

Similarly, PEtab-select is now accessed through `PEtabSelect.jl`:

```julia
# new
using PEtabSelect
petab_select(path_yaml, IPNewton(); nmultistarts=10)
# old
using PyCall
ENV["PYTHON"] = "path_to_python"
import Pkg; Pkg.build("PyCall")
petab_select(path_yaml, IPNewton(); nmultistarts=10)
```

### Plotting

It is now possible to plot the residuals, and standardized (normalized by standard
deviation) residuals:

```julia
plot(x, prob; plot_type = :residuals)
plot(x, prob; plot_type = :standardized_residuals)
```

### PEtab format v2 support

PEtab.jl v4 adds PEtab v2 support while maintaining compatibility with v1. In PEtab v2,
simulations are organized into experiments, not conditions. For PEtab v2 problems, `get_`
and `plot` use the `experiment` keyword instead of `condition`, e.g.
`get_u0(...; experiment=:e1)` and `plot(x, prob; experiment=:e1)`.

### Printing and `describe`

Printing for PEtab structs (e.g. `PEtabODEProblem`, `PEtabModel`) was updated. The
`describe` function was added to summarize key statistics for a `PEtabODEProblem`.

## PEtab 3.11.0

Update to ModelingToolkit.jl version 9.84.

## PEtab 3.10.0

Added support for optional `rng::AbstractRNG` argument to `get_startguesses` and
`calibrate_multistart` for reproducible sampling of starting points. If omitted,
`Random.default_rng()` is used, so existing call signatures still work:

```julia
x_start = get_startguesses(prob, 10)
ms_res = calibrate_multistart(petab_prob, IPNewton(), 50)
```

Wherever, with this update, an `rng` can be explicitly provided as the first argument:

```julia
rng = Random.Xoshiro(1)
x_start = get_startguesses(rng, prob, 10)
ms_res = calibrate_multistart(rng, petab_prob, IPNewton(), 50)
```

## PEtab 3.9.0

Updating plotting functionality. The waterfall plot will now, based on the magnitude of
likelihood values, decide between a log or linear scale. Moreover, an automatic re-scaling
of the y-axis is now applied if the waterfall plot should be on log-scale, and there are
negative likelihood values.

## PEtab 3.8.0

Update to use Catalyst.jl version 15.

## PEtab 3.7.0

Add the `get_system` utility function. This function retrieves the `PEtabODEProblem` model
system (either a `ReactionSystem` or an `ODESystem`) along with its associated species and
parameter maps for any chosen simulation condition.

## PEtab 3.6.0

Update petab-select to version 0.3. This only changes internals, the user interface for
PEtab-select remains unchanged.

## PEtab 3.5.0

Add `log2` transformation for parameters and observable transformations.

## PEtab 3.4.0

Plot updates:

- Makes it possible to provide a parameter vector instead of only a parameter estimation
  results when plotting model fit.
- Add option to change label in model fit plot to show observable id.

## PEtab 3.3.0

Update to Optimization.jl v4 and DiffEqCallbacks v4.

## PEtab 3.2.0

Added support for running multi-start parameter estimation with `calibrate_multistart` in
parallel using `pmap` from Distributed.jl via the `nprocs` keyword. For example, to now run
parameter estimation with two processes in parallel, use:

```julia
ms_res = calibrate_multistart(petab_prob, IPNewton(), 50; nprocs = 2)
```

## PEtab 3.1.0

Added support for truncated priors for `PEtabParameter`. Before this update the following
would yield an error:

```julia
pk1 = PEtabParameter(:k1; prior = truncated(Normal(1.0, 1.0), 0.0, 3.0))
```

But now it works as expected.

## PEtab 3.0.0

This version is a breaking release prompted by the update of ModelingToolkit to v9 and
Catalyst to v14. Along with updating these packages, PEtab.jl also underwent a major update
to make the package easier to use. The major changes are:

- A major rewriting of the documentation to make it more accessible. References and more
  details on the math behind PEtab.jl have also been added to the documentation.
- A near-complete refactoring of the code base to improve maintainability. As a result, any
  code relying on PEtab.jl internals will likely no longer work.
- Renaming of functions and function arguments to better align with the naming convention in
  the Julia SciML ecosystem.
- Added support for `ComponentArray` for parameter estimation, making it easier to interact
  with both the input and output when doing parameter estimation.
- Dropping Zygote.jl support. In previous versions, PEtab.jl supported gradients via
  `Zygote.gradient` on the objective function. However, this was the slowest gradient method
  by far, and the hardest to maintain, so it has been removed.

## Changes in defining models

Following the update to ModelingToolkit v9, the syntax for defining models as `ODESystem`
has changed. In particular, a model that was previously defined as:

```julia
using ModelingToolkit
@parameters c1, c2, c3, S0
@variables t S(t) SE(t) P(t) E(t)
D = Differential(t)
eqs = [
    D(S) ~ -c1*S*E + c2*SE,
    D(E) ~ -c1*S*E + c2*SE + c3*SE,
    D(SE) ~ c1*S*E - c2*SE - c3*SE,
    D(P) ~ c3*SE
]
@named sys = ODESystem(eqs; defaults=Dict(S => S0, c3 => 1.0, E => 50.0))
```

should now be defined as:

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
@mtkmodel SYS begin
    @parameters begin
        S0
        c1
        c2
        c3 = 1.0
    end
    @variables begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.0
        P(t) = 0.0
    end
    @equations begin
        D(S) ~ -c1 * S * E + c2 * SE
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE
        D(P) ~ c3 * SE
    end
end
@mtkbuild sys = SYS()
```

Besides the syntax change, we now recommend setting any initial values directly in the model
formulation instead of via the default keyword. Similarly, following the update to Catalyst
v14, a model that was previously defined as:

```julia
using Catalyst
rn = @reaction_network begin
    @parameters se0
    @species SE(t) = se0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]
parametermap = [:c3 => 1.0]
```

should now be defined as:

```julia
using Catalyst
t = default_t()
rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]
```

For more information on changes in how to define models, see the Catalyst documentation and
the ModelingToolkit documentation.

## Renaming

In PEtab.jl v3, several functions were renamed to better align with the naming convention in
the Julia SciML ecosystem. Additionally, a subset of keyword arguments when creating
`PEtabODEProblem` and `PEtabModel`, as well as some field names in `PEtabODEProblem`, have
been changed. Moreover, neither `PEtabModel` nor `PEtabODEProblem` prints progress by
default anymore when building. For an up-to-date list, see the API section in the
documentation. Below is a summary.

### Renaming of functions

The following functions have been renamed:

- `calibrate_model` -> `calibrate`
- `calibrate_model_multistart(petab_prob, alg, nmultistarts, dirsave; kwargs...)` ->
  `calibrate_multistart(petab_prob, alg, nmultistarts; kwargs...)`
  - Note: `dirsave` is now an optional keyword argument.
- `run_PEtab_select` -> `petab_select`
- `generate_startguesses` -> `get_startguesses`
- `remake_PEtab_problem` -> `remake`
- `PEtabMultistartOptimisationResult` -> `PEtabMultistartResult`

The get functions (`get_u0`, `get_ps`, `get_odeproblem`, and `get_odesol`) have also had a
keyword argument renamed, specifically `condition_id` is now provided via `cid`. For more
details, see the API section in the documentation.

### Renaming in `PEtabODEProblem`

The `PEtabODEProblem` struct has also been updated with new field names. In previous
versions, the unknown parameter vector was often referred to as `θ`. Now, the unknown
parameter vector to estimate is generally referred to as `x`. Additionally, the following
fields in `PEtabODEProblem` have been renamed:

- `prob.compute_cost` -> `prob.nllh`
  - To better reflect that the objective function in PEtab.jl is the
    negative-log-likelihood.
- `prob.compute_gradient!` -> `prob.grad!`
- `prob.compute_gradient` -> `prob.grad`
- `prob.compute_hessian!` -> `prob.hess!`
- `prob.compute_hessian` -> `prob.hess`
- `prob.compute_chi2` -> `prob.chi2`
- `prob.compute_simulated_values` -> `prob.simulated_values`
- `prob.compute_residuals` -> `prob.residuals`
- `prob.θnames` -> `prob.xnames`

As noted above, `compute` has been dropped from most functions that the `PEtabODEProblem`
creates. Additionally, when creating the `PEtabODEProblem`, the following keyword arguments
have been renamed:

- `ode_solver` -> `odesolver`
- `ode_solver_gradient` -> `odesolver_gradient`

### Renaming in `PEtabModel`

For creating a `PEtabModel` in Julia, the following keyword arguments have been renamed:

- `state_map` -> `speciemap`
- `parameter_map` -> `parametermap`

Moreover, previously `PEtabModel` had two constructors:

```julia
PEtabModel(sys, simulation_conditions, observables, measurements, parameters; kwargs...)
PEtabModel(sys, observables, measurements, parameters; kwargs...)
```

The first constructor was used if the model had any simulation conditions. This constructor
has now been dropped, and the only valid constructor is:

```julia
PEtabModel(sys, observables, measurements, parameters;
           simulation_conditions = simulation_conditions kwargs...)
```

where `simulation_conditions` is now an optional keyword argument.

## New functions

PEtab.jl has also introduced a new utility function (more information can be found in the
documentation):

- `get_x`: Retrieves the nominal parameter vector, with parameters arranged in the correct
  order expected by a `PEtabODEProblem` for parameter estimation/inference.

## Adding support for ComponentArrays.jl

In previous versions of PEtab.jl, the input to the functions generated by the
`PEtabODEProblem`, as well as functions for parameter estimation (e.g., `calibrate`), was
expected to be a Julia `Vector`, as the objective and derivative functions created by the
`PEtabODEProblem` required a `Vector` input. Since the `PEtabODEProblem` expected the
parameters in this input `Vector` to be in a specific order, interacting with the parameter
vector became inconvenient, as users had to track both parameter values and names, with
names stored separately. Additionally, the default estimation of parameters on the `log10`
scale often caused confusion, as this was not apparent with a `Vector` input.

To address these issues, the functions generated by the `PEtabODEProblem`, as well as
functions for parameter estimation (e.g., `calibrate`), now support and default to using
`ComponentArray`, which is essentially a named vector that stores both the name and value of
each parameter. Specifically, previously, PEtab.jl expected input in the form:

```julia
x = petab_prob.θ_nominalT
[0.0, 2.0, 3.0]
```

for `calibrate` or when for example computing the objective function, where the parameter
names and expected order were stored in a separate vector (`xnames`). Now, it accepts input
in the form:

```julia
x = get_x(petab_prob)
ComponentVector{Float64}(log10_c1 = 0.0, c2 = 2.0, c3 = 3.0)
```

where a potential prefix (e.g., `log10`) specifies the parameter scale (note that the
`ComponentArray` must still have parameters with the correct scale in the specified order,
but this is automatically handled by functions like `get_x` and `get_startguesses`). In
addition to making it easier to identify the parameter scale, this also simplifies
interacting with and modifying parameter vectors. For example, to change `c2` in the
`ComponentArray`, you can simply do `x[:c2] = 3.0` instead of the previous `x[ic2] = 2.0`,
where the index had to be identified using the vector of names.

In summary, the functions related to parameter estimation in PEtab.jl (e.g., functions
generated by `PEtabODEProblem` as well as functions like `calibrate`) now support
`ComponentArray`. However, all functions still support `Vector` input as well.
