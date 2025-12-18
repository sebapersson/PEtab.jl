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
    PEtabObservable(observable_formula, noise_formula; distribution = Normal)

Observation model linking model output (`observable_formula`) to measurement data via a
likelihood defined by `distribution` with noise/scale given by `noise_formula`.

For any supported likelihood `distribution`, `observable_formula` is the distribution
median, i.e., measurements are assumed equally likely to lie above or below the model output.

## Arguments

- `observable_formula`: Observable expression (`String` or `Symbolics` equation) supporting
    standard Julia functions (e.g. `exp`, `log`, `sin`, `cos`). Variables in the formula
    must be model species, model parameters, or a `PEtabParameter`. May include
    time-point-specific observable parameters (see documentation).
- `noise_formula`: Noise/scale expression (String or Symbolics equation), same rules as
    `observable_formula`. May include time-point-specific noise parameters (see documentation).

## Keyword Arguments

- `distribution`: Distribution of the measurement noise. Valid options are `Normal`,
    `Laplace`, `LogNormal`, and `LogLaplace`. See below for mathematical definition.

## Mathematical definition

For a measurement `m`, model output `y = observable_formula`, and a noise parameter
`σ = noise_formula`, `PEtabObservable` defines the likelihood linking the model output to
the measurement data: ``\\pi(m \\mid y, \\sigma)``.

For `distribution = Normal`, the measurement is assumed to be normally distributed with
``m \\sim \\mathcal{N}(y, \\sigma^2)``. The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}}\\mathrm{exp}\\bigg( -\\frac{(m - y)^2}{2\\sigma^2} \\bigg)
````

If `\\sigma = 1`, this likelihood reduces to the least-squares objective function.

For `distribution = Laplace`, the measurement is assumed to be Laplace distributed with
`m \\sim \\mathcal{L}(y, \\sigma)`. The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{2\\sigma}\\mathrm{exp}\\bigg( -\\frac{|m - y|}{\\sigma} \\bigg)
```

For `distribution = LogNormal`, the log of the measurement is assumed to be Normal
distributed with `\\mathrm{log}(m) \\sim \\mathcal{N}(\\mathrm{log}(y), \\sigma^2)`
(requires `m > 0` and `y > 0`). The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}\\, m}\\mathrm{exp}\\bigg( -\\frac{\\big(\\mathrm{log}(m) - \\mathrm{log}(y)\\big)^2}{2\\sigma^2} \\bigg)
```

For `distribution = LogLaplace`, the log of the measurement is assumed to be Laplace
distributed with `\\mathrm{log}(m) \\sim \\mathcal{L}(\\mathrm{log}(y), \\sigma)`
(requires `m > 0` and `y > 0`). The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{2\\sigma\\, m}\\mathrm{exp}\\bigg( -\\frac{\\big|\\mathrm{log}(m) - \\mathrm{log}(y)\\big|}{\\sigma} \\bigg)
```

For numerical stability, PEtab.jl works with the log-likelihood in practice.
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
    PEtabCondition(condition_id, target_ids, target_values; t0 = 0.0)

A simulation condition that overrides `target_ids` with `target_values` under `condition_id`.

Used to set control parameters for different experimental conditions. For examples, see
the online documentation.

## Arguments

- `condition_id::Union{String, Symbol}`: Simulation condition identifier.
- `target_id`: Entity id or ids to override: a single id or a `Vector` of ids.
    Ids (provided as `String` or `Symbol`) may be model specie ids and/or model parameter
    ids for model parameters that are not estimated.
- `target_value`: Value/expression assigned to `target_id`. A `String` or a `Symbolics`
    expression which may use standard Julia functions (e.g. `exp`, `log`, `sin`).
    Must match the length of `target_id` when `target_id` is a vector. Any variables
    referenced must be model parameters or `PEtabParameter`(s) (specie
    variables are not allowed).

## Keyword Arguments

- `t0`: Model simulation start time for `condition_id` (defaults to `0.0`).
"""
struct PEtabCondition
    condition_id::String
    target_ids::Vector{String}
    target_values::Vector{String}
    t0::Float64
end
function PEtabCondition(condition_id::UserFormula, target_id::UserFormula, target_value::Union{UserFormula, Real}; t0::Real = 0.0)
    PEtabCondition(string(condition_id), [string(target_id)], [string(target_value)], t0)
end
function PEtabCondition(condition_id::UserFormula, target_ids::AbstractVector, target_values::AbstractVector; t0::Real = 0.0)
    if length(target_ids) != length(target_values)
        throw(PEtabFormatError("For condition $(condition_id), the number of target ids \
            ($(length(target_ids))) must equal the number of target values \
            ($(length(target_values)))."))
    end
    return PEtabCondition(string(condition_id), string.(target_ids), string.(target_values), t0)
end

"""
    PEtabEvent(condition, target_ids, target_values; condition_ids = [:all])

Model event triggered when `condition` transitions from `false` to `true`, setting the value
of `target_ids` to `target_values`.

For examples, see the online documentation.

## Arguments
- `condition`: A Boolean expression that triggers the event when it transitions from
    `false` to `true`. For example, if `t == c1`, the event is triggered when the model
    time `t` equals the value of model parameter `c1`. For `S > 2.0`, the event triggers
    when model specie `S` passes 2.0 from below.
- `target_ids`: Entity id or ids to assign value a, either a single id or a `Vector` of ids.
    Ids (provided as `String` or `Symbol`) may be model specie ids and/or model parameters.
- `target_values`: Value/expression assigned to `target_id`. A `String` or a `Symbolics`
    equation which may use standard Julia functions (e.g. `exp`, `log`, `sin`).
    Must match the length of `target_id` when `target_id` is a vector. Any variables
    referenced must be model parameters or model species.
- `condition_ids` (optional): Simulation condition(s) ids (provided as `String` or `Symbol`)
    to which the event applies. If set to `[:all]` (default), the event is applied to all
    simulation conditions.
"""
struct PEtabEvent
    condition::String
    target_ids::Vector{String}
    target_values::Vector{String}
    trigger_time::Float64
    condition_ids::Vector{Symbol}
end
function PEtabEvent(condition::Union{UserFormula, Real}, target_ids::UserFormula, target_values::Union{UserFormula, Real}; trigger_time::Real = NaN, condition_ids::Union{Vector{String}, Vector{Symbol}} = Symbol[])
    return PEtabEvent(string(condition), [string(target_ids)], [string(target_values)], trigger_time, Symbol.(condition_ids))
end
function PEtabEvent(condition::Union{UserFormula, Real}, target_ids::AbstractVector, target_values::AbstractVector; trigger_time::Real = NaN, condition_ids::Union{Vector{String}, Vector{Symbol}} = Symbol[])
    if length(target_ids) != length(target_values)
        throw(PEtabFormatError("For a PEtabEvent, the number of target ids \
            ($(length(target_ids))) must equal the number of target values \
            ($(length(target_values)))."))
    end
    return PEtabEvent(string(condition), string.(target_ids), string.(target_values), trigger_time, Symbol.(condition_ids))
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
    sys_observables::Dict{Symbol, Function}
end
