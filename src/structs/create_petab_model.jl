"""
    PEtabParameter(x; kwargs...)

Parameter estimation information for parameter `x`.

All pest to be estimated in a `PEtabODEProblem` must be declared as a
`PEtabParameter`, and `x` must be the name of a parameter that appears in the model,
observable formula, or noise formula.

## Keyword Arguments

- `lb::Float64 = 1e-3`: The lower parameter bound for parameter estimation.
- `ub::Float64 = 1e3`: The upper parameter bound for parameter estimation.
- `scale::Symbol = :log10`: The scale on which to estimate the parameter. Allowed options
    are `:log10` (default), `:log`, and `:lin`. Estimating pest on the `log10` scale
    typically improves performance and is recommended.
- `prior = nothing`: An optional continuous univariate prior distribution from
    [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).
- `prior_on_linear_scale = true`: Whether the prior is on the linear scale (default) or on
    the transformed scale. For example, if `scale = :log10` and
    `prior_on_linear_scale = false`, the prior acts on the transformed value; `log10(id)`.
- `estimate = true`: Whether the parameter should be estimated (default) or treated as a
    constant.
- `value = nothing`: Value to use if `estimate = false`. Defaults to the midpoint between
    `lb` and `ub`.

## Description

If a prior ``\\pi(x_i)`` is provided, the parameter estimation problem becomes a maximum a
posteriori problem instead of a maximum likelihood problem. Practically, instead of
minimizing the negative log-likelihood,``-\\ell(x)``, the negative posterior is minimized:
```math
\\min -\\ell(x) - \\sum_{i} π(xᵢ)
```
For all pest ``x_{i}`` with a prior.

## Example

```julia
# Parameter with a LogNormal prior
using PEtab, Distributions
PEtabParameter(:c1; prior=LogNormal(3.0, 1.0))
```
"""
struct PEtabParameter
    parameter::Union{Num, Symbol}
    estimate::Bool
    value::Union{Nothing, Float64}
    lb::Union{Nothing, Float64}
    ub::Union{Nothing, Float64}
    prior::Union{Nothing, Distribution{Univariate, Continuous}}
    prior_on_linear_scale::Bool
    scale::Union{Nothing, Symbol} # :log10, :linear and :log supported.
    sample_prior::Bool
end
function PEtabParameter(id::Union{Num, Symbol};
                        estimate::Bool = true,
                        value::Union{Nothing, Float64} = nothing,
                        lb::Union{Nothing, Float64} = 1e-3,
                        ub::Union{Nothing, Float64} = 1e3,
                        prior::Union{Nothing, Distribution{Univariate, Continuous}} = nothing,
                        prior_on_linear_scale::Bool = true,
                        scale::Union{Nothing, Symbol} = :log10,
                        sample_prior::Bool = true)
    return PEtabParameter(id, estimate, value, lb, ub, prior, prior_on_linear_scale, scale,
                          sample_prior)
end

"""
    PEtabObservable(obs_formula, noise_formula; kwargs...)

Formulas defining the likelihood that links the model output to the measurement data.

`obs_formula` describes how the model output relates to the measurement data, while
`noise_formula` describes the measurement error and can be a constant numerical value.
The observable and noise formulas can be any valid Julia equation (see example 2 below).
Variables used in these formulas must be either model species, model pest, or
pest defined as `PEtabParameter`. The formulas can also include time-point-specific
noise and observable pest; for more information on this see the documentation.

## Keyword Arguments

- `transformation`: Specifies the transformation applied to the observable and measurement.
    Valid options are `:lin` (normal measurement noise), `:log`, or `:log10` (log-normal
    measurement noise). See below for more details.

## Description

For a measurement ``y``, an observable ``h = obs\\_formula``, and a standard deviation
``\\sigma = noise\\_formula``, the `PEtabObservable` defines the likelihood that links the
model output to the measurement data: ``\\pi(y \\mid h, \\sigma)``. For
`transformation = :lin`, the measurement noise is assumed to be normally distributed, and
the likelihood is given by:
```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}}\\mathrm{exp}\\bigg( -\\frac{y - h}{2\\sigma^2} \\bigg)
```

As a special case, if ``\\sigma = 1``, this likelihood reduces to the least-squares
objective function. For `transformation = :log`, the measurement noise is assumed to be
log-normally distributed, and the likelihood is given by:
```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2 y^2}}\\mathrm{exp}\\bigg( -\\frac{\\mathrm{log}(y) - \\mathrm{log}(h)}{2\\sigma^2} \\bigg)
```

For `transformation = :log10`, the measurement noise is assumed to be log10-normally
distributed, and the likelihood is given by:
```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2 y^2}\\mathrm{log}(10) }\\mathrm{exp}\\bigg( -\\frac{\\mathrm{log}_{10}(y) - \\mathrm{log}_{10}(h)}{2\\sigma^2} \\bigg)
```

It should be noted, that in practice when running parameter estimation or Bayesian
inference, PEtab.jl uses the log-likelihood for numerical stability.

## Examples
```julia
# Example 1: Constant known measurement noise σ=3.0
@unpack X = rn  # 'rn' is the dynamic model and X is assumed model specie
PEtabObservable(X, 3.0, transformation=:log)
```
```julia
# Example 2: Unknown measurement noise σ (which must be defined as PEtabParameter)
@unpack X, Y = rn  # 'rn' is the dynamic model and X and Y are assumed model species
@pest sigma
PEtabObservable((X + Y) / X, sigma)
```
"""
struct PEtabObservable
    obs::Any
    transformation::Union{Nothing, Symbol}
    noise_formula::Any
end
function PEtabObservable(obs_formula, noise_formula;
                         transformation::Symbol = :lin)::PEtabObservable
    return PEtabObservable(obs_formula, transformation, noise_formula)
end

"""
    PEtabEvent(condition, affects, targets)

A model event triggered by `condition` that sets the value of `targets` to that of
`affects`.

For a collection of examples with plots, see the documentation.

## Arguments
- `condition`: A Boolean expression that triggers the event when it transitions from
    `false` to `true`. For example, if `t == c1`, the event is triggered when the model
    time `t` equals `c1`. For `S > 2.0`, the event triggers when `S` passes the value 2.0
    from below.
- `affects`: An algebraic expression consisting of model species and pest that
    describes the effect of the event. It can be a single expression or a vector of
    multiple effects.
- `targets`: Model species or pest that the event acts on. Must match the dimension of
    `affects`.

## Examples
```julia
using Catalyst, PEtab
# Trigger event at t = 3.0, and update A <- A + 5
rn = @reaction_network begin
    (k1, k2), A <--> B
end
@unpack A = rn
t = default_time()
event = PEtabEvent(3.0 == 3, A + 5.0, A)
```
```julia
using Catalyst
# Trigger event when A == 0.2, and update B <- 2.0
rn = @reaction_network begin
    (k1, k2), A <--> B
end
@unpack A, B = rn
event = PEtabEvent(A == 0.2, 2.0, B)
```
"""
struct PEtabEvent
    condition
    affect
    target
end

"""
    PEtabModel(sys, observables::Dict{String, PEtabObservable}, measurements::DataFrame,
               parameters::Vector{PEtabParameter}; kwargs...)

From a `ReactionSystem` or an `ODESystem` model, `observables` that link the model to
`measurements` and `parameters` to estimate, create a `PEtabModel` for parameter estimation.

For examples on how to create a `PEtabModel`, see the documentation.

See also [`PEtabObservable`](@ref), [`PEtabParameter`](@ref), and [`PEtabEvent`](@ref).

## Keyword Arguments

- `simulation_conditions = nothing`: An optional dictionary specifying initial specie values
    and/or model parameters for each simulation condition. Required if the model has
    multiple simulation conditions.
- `events = nothing`: Optional model events (callbacks) provided as `PEtabEvent`. Multiple
    events should be provided as a Vector of `PEtabEvent`.
- `verbose::Bool = false`: Whether to print progress while building the `PEtabModel`.


    PEtabModel(path_yaml; kwargs...)

Import a PEtab problem in the standard format, with the YAML file located at `path_yaml`,
into a `PEtabModel` for parameter estimation.

For examples on how to import a PEtab problem, see the documentation.

## Keyword Arguments
- `ifelse_to_callback::Bool = true`: Rewrites `ifelse` (SBML piecewise) expressions to use
  [callbacks](https://github.com/SciML/DiffEqCallbacks.jl). This improves simulation
  runtime. It is strongly recommended to set this to `true`.
- `verbose::Bool = false`: Whether to print progress while building the `PEtabModel`.
- `write_to_file::Bool = false`: Whether to write the generated Julia functions to files in
   the same directory as the PEtab problem. Useful for debugging.
"""
struct PEtabModel
    name::String
    h::Function
    u0!::Function
    u0::Function
    sd::Function
    float_tspan::Bool
    paths::Dict{Symbol, String}
    sys::Any
    sys_mutated::Any
    parametermap::Any
    statemap::Any
    petab_tables::Dict{Symbol, DataFrame}
    callbacks::SciMLBase.DECallback
    defined_in_julia::Bool
end
