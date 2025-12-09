"""
    PEtabParameter(x; kwargs...)

Parameter estimation information for parameter `x`.

All parameters to be estimated in a `PEtabODEProblem` must be declared as a
`PEtabParameter`, and `x` must be the name of a parameter that appears in the model,
observable formula, or noise formula.

## Keyword Arguments

- `lb::Float64 = 1e-3`: The lower parameter bound for parameter estimation. Must
    be specified on the linear scale. For example, if `scale = :log10`, provide the
    bound as `1e-3` rather than `log10(1e-3)`.
- `ub::Float64 = 1e3`: The upper parameter bound for parameter estimation. Must as for
    `lb` be provided on linear scale.
- `scale::Symbol = :log10`: The scale on which to estimate the parameter. Allowed options
    are `:log10` (default), `:log2` `:log`, and `:lin`. Estimating on the `log10`
    scale typically improves performance and is recommended.
- `prior = nothing`: An optional continuous univariate parameter prior distribution from
    [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). The prior
    overrides any parameter bounds.
- `prior_on_linear_scale = true`: Whether the prior is on the linear scale (default) or on
    the transformed scale. For example, if `scale = :log10` and
    `prior_on_linear_scale = false`, the prior acts on the transformed value; `log10(x)`.
- `estimate = true`: Whether the parameter should be estimated (default) or treated as a
    constant.
- `value = nothing`: Value to use if `estimate = false`, and value retreived by the `get_x`
    function. Defaults to the midpoint between `lb` and `ub`.

## Description

If a prior ``\\pi(x_i)`` is provided, the parameter estimation problem becomes a maximum a
posteriori problem instead of a maximum likelihood problem. Practically, instead of
minimizing the negative log-likelihood,``-\\ell(x)``, the negative posterior is minimized:

```math
\\min_{\\mathbf{x}} -\\ell(\\mathbf{x}) - \\sum_{i} \\pi(x_i)
```

For all parameters ``i`` with a prior.
"""
struct PEtabParameter
    parameter::Union{Num, Symbol}
    estimate::Bool
    value::Union{Nothing, Float64}
    lb::Union{Nothing, Float64}
    ub::Union{Nothing, Float64}
    prior::Union{Nothing, Distribution{Univariate, Continuous}}
    prior_on_linear_scale::Bool
    scale::Symbol
    sample_prior::Bool
end
function PEtabParameter(id::Union{Num, Symbol}; estimate::Bool = true,
                        value::Union{Nothing, Float64} = nothing, sample_prior::Bool = true,
                        lb::Union{Nothing, Float64} = 1e-3,
                        ub::Union{Nothing, Float64} = 1e3,
                        prior::Union{Nothing, Distribution{Univariate, Continuous}} = nothing,
                        prior_on_linear_scale::Bool = true, scale::Symbol = :log10)
    return PEtabParameter(id, estimate, value, lb, ub, prior, prior_on_linear_scale, scale,
                          sample_prior)
end

"""
    PEtabObservable(obs_formula, noise_formula; kwargs...)

Formulas defining the likelihood that links the model output to the measurement data.

`obs_formula` describes how the model output relates to the measurement data, while
`noise_formula` describes the standard deviation (measurement error) and can be an equation
or a numerical value. Both the observable and noise formulas can be a valid Julia equation.
Variables used in these formulas must be either model species, model parameters, or
parameters defined as `PEtabParameter`. The formulas can also include time-point-specific
noise and observable parameters; for more information, see the documentation.

## Keyword Argument

- `transformation`: The transformation applied to the observable and its corresponding
    measurements. Valid options are `:lin` (normal measurement noise), `:log`, `log2` or
    `:log10` (log-normal measurement noise). See below for more details.

## Description

For a measurement `y`, an observable `h = obs_formula`, and a standard deviation
`σ = noise_formula`, the `PEtabObservable` defines the likelihood that links the
model output to the measurement data: ``\\pi(y \\mid h, \\sigma)``. For
`transformation = :lin`, the measurement noise is assumed to be normally distributed, and
the likelihood is given by:

```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}}\\mathrm{exp}\\bigg( -\\frac{(y - h)^2}{2\\sigma^2} \\bigg)
```

As a special case, if ``\\sigma = 1``, this likelihood reduces to the least-squares
objective function. For `transformation = :log`, the measurement noise is assumed to be
log-normally distributed, and the likelihood is given by:
```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2 y^2}}\\mathrm{exp}\\bigg( -\\frac{\\big(\\mathrm{log}(y) - \\mathrm{log}(h)\\big)^2}{2\\sigma^2} \\bigg)
```

For `transformation = :log10`, the measurement noise is assumed to be log10-normally
distributed, and the likelihood is given by:
```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2 y^2}\\mathrm{log}(10) }\\mathrm{exp}\\bigg( -\\frac{\\big(\\mathrm{log}_{10}(y) - \\mathrm{log}_{10}(h)\\big)^2}{2\\sigma^2} \\bigg)
```

Lastly, for `transformation = :log2`, the measurement noise is assumed to be log2-normally
distributed, and the likelihood is given by:
```math
\\pi(y|h, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2 y^2}\\mathrm{log}(2) }\\mathrm{exp}\\bigg( -\\frac{\\big(\\mathrm{log}_{2}(y) - \\mathrm{log}_{2}(h)\\big)^2}{2\\sigma^2} \\bigg)
```

For numerical stabillity, PEtab.jl works with the log-likelihood in practice.
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

For a collection of examples with corresponding plots, see the documentation.

## Arguments
- `condition`: A Boolean expression that triggers the event when it transitions from
    `false` to `true`. For example, if `t == c1`, the event is triggered when the model
    time `t` equals `c1`. For `S > 2.0`, the event triggers when specie `S` passes 2.0
    from below.
- `affects`: An equation of of model species and parameters that describes the effect of
    the event. It can be a single expression or a vector if there are multiple targets.
- `targets`: Model species or parameters that the event acts on. Must match the dimension
    of `affects`.
"""
struct PEtabEvent
    condition::Any
    affect::Any
    target::Any
    trigger_time::Float64
    condition_ids::Vector{Symbol}
end
function PEtabEvent(condition, affect, target; trigger_time = Inf, conditions_ids::Union{Vector{String}, Vector{Symbol}} = Symbol[])
    return PEtabEvent(condition, affect, target, trigger_time, Symbol.(conditions_ids))
end
function PEtabEvent(condition_event_df::DataFrame, trigger_time::Real, simulation_condition_id::String)
    targets = condition_event_df.targetId
    condition = "t == $(trigger_time)"
    affects = fill("", length(targets))
    for (i, target_value) in pairs(condition_event_df.targetValue)
        affects[i] = string(target_value)
    end
    return PEtabEvent(condition, affects, targets, trigger_time, [Symbol(simulation_condition_id)])
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
    and/or model parameters for each simulation condition. Only required if the model has
    multiple simulation conditions.
- `events = nothing`: Optional model events (callbacks) provided as `PEtabEvent`. Multiple
    events should be provided as a `Vector` of `PEtabEvent`.
- `verbose::Bool = false`: Whether to print progress while building the `PEtabModel`.


    PEtabModel(path_yaml; kwargs...)

Import a PEtab problem in the standard format with YAML file at `path_yaml` into a
`PEtabModel` for parameter estimation.

For examples on how to import a PEtab problem, see the documentation.

## Keyword Arguments
- `ifelse_to_callback::Bool = true`: Whether to rewrite `ifelse` (SBML piecewise)
    expressions to [callbacks](https://github.com/SciML/DiffEqCallbacks.jl). This improves
    simulation runtime. It is strongly recommended to set this to `true`.
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
    speciemap::Any
    petab_tables::Dict{Symbol, DataFrame}
    callbacks::Dict{Symbol, SciMLBase.DECallback}
    defined_in_julia::Bool
    petab_events::Vector{PEtabEvent}
end
