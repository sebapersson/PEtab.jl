"""
    PEtabParameter(parameter_id; kwargs...)

Parameter-estimation data for parameter `parameter_id` (bounds, scale, prior, and whether
to estimate).

All parameters estimated in a `PEtabODEProblem` must be declared as `PEtabParameter`.
`parameter_id` must correspond to a model parameter and/or a parameter appearing in an
`observable_formula` or `noise_formula` of a `PEtabObservable`.

# Keyword Arguments
* `scale::Symbol = :log10`: Scale the parameter is estimated on. One of `:log10` (default),
    `:log2`, `:log`, or `:lin`. Estimating on a log scale often improves performance and is
    recommended.
- `lb`: Lower bound, specified on the **linear** scale; e.g. with `scale = :log10`, pass
    `lb = 1e-3`, not `log10(1e-3)`. Defaults to `1e-3` without a `prior`, otherwise to the
    lower bound of the prior support.
- `ub`: Upper bound, same convention as `lb`. Defaults to `1e3` without a `prior`,
    otherwise to the upper bound of the prior support.
- `prior = nothing`: Optional prior distribution acting on the **linear** parameter scale
    (i.e. even if `scale = :log10`, the prior is on `x`, not on `log10(x)`). If the prior’s
    support extends beyond provided `lb/ub` bounds, it is truncated by `[lb, ub]`. Any
    continuous univariate distribution from
    [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) is supported,
    including truncated distributions.
- `estimate::Bool = true`: Whether the parameter is estimated (default `true`) or treated
    as a constant (`false`).
- `value = nothing`: Value used when `estimate = false`, and the value returned by
    `get_x`. Defaults to the midpoint of `[lb, ub]`.

# Priors and Parameter Estimation

If at least one parameter in a `PEtabODEProblem` has a `prior` specified, parameter
estimation uses a maximum-a-posteriori (MAP) objective:

```math
\\min_{\\mathbf{x}} -\\ell(\\mathbf{x}) - \\sum_{i \\in \\mathcal{I}} \\log \\pi_i(x_i)
```

where ``\\mathcal{I}`` indexes parameters with an explicit prior density ``\\pi_i``.
If no parameter has a prior, parameter estimation reduces to the maximum-likelihood (ML)
objective ``-\\ell(\\mathbf{x})``.
"""
struct PEtabParameter
    parameter_id::String
    estimate::Bool
    value::Union{Nothing, Float64}
    lb::Union{Nothing, Float64}
    ub::Union{Nothing, Float64}
    prior::Union{Nothing, Distribution{Univariate, Continuous}}
    scale::Symbol
    sample_prior::Bool
end
function PEtabParameter(parameter_id::UserFormula; estimate::Bool = true,
                        value::Union{Nothing, Float64} = nothing, sample_prior::Bool = true,
                        lb::Union{Nothing, Real} = nothing, ub::Union{Nothing, Real} = nothing,
                        prior::Union{Nothing, Distribution{Univariate, Continuous}} = nothing,
                        scale::Symbol = :log10)
    if isnothing(prior)
        lb = isnothing(lb) ? 1e-3 : lb
        ub = isnothing(ub) ? 1e3 : ub
    end

    if !isnothing(prior)
        prior_support = Distributions.support(prior)
        lb = isnothing(lb) ? prior_support.lb : lb
        ub = isnothing(ub) ? prior_support.ub : ub

        if lb > prior_support.lb || ub < prior_support.ub
            prior = truncated(prior, lb, ub)
        end
    end

    if isnothing(value)
        if any(isinf.(abs.([lb, ub])))
            value = 1.0
        else
            value = (lb + ub) .* 0.5
        end
    end

    return PEtabParameter(string(parameter_id), estimate, value, lb, ub, prior, scale, sample_prior)
end

"""
    PEtabObservable(observable_id, observable_formula, noise_formula; distribution = Normal)

Observation model linking model output (`observable_formula`) to measurement data via a
likelihood defined by `distribution` with noise/scale given by `noise_formula`.

For examples, see the online package documentation.

# Arguments

- `observable_id::Union{String, Symbol}`: Observable identifier. Measurements
    are linked to this observable via the column `obs_id` in the measurement table.
- `observable_formula`: Observable expression. Two supported forms:
    - `Model-observable`: A `Symbol` identifier matching an observable defined in a Catalyst
      `ReactionSystem` `@observables` block, or a non-differential variable defined in a
      ModelingToolkit `ODESystem` `@variables` block.
    - `expression`: A `String`, `:Symbol`, `Real`, or a Symbolics expression (`Num`). Can
      include  standard Julia functions (e.g. `exp`, `log`, `sin`, `cos`). Variables may
      reference model species, model parameters, or `PEtabParameter`s. Can include
      time-point-specific parameters (see documentation for examples).
- `noise_formula`: Noise/scale expression (String or Symbolics equation), same rules as
    `observable_formula`. May include time-point-specific parameters (see documentation).

# Keyword Arguments

- `distribution`: Measurement noise distribution. Valid options are `Normal` (default),
    `Laplace`, `LogNormal`, `Log2Normal`, `Log10Normal` and `LogLaplace`. See below for
    mathematical definition.

# Mathematical description

For a measurement `m`, model output `y = observable_formula`, and a noise parameter
`σ = noise_formula`, `PEtabObservable` defines the likelihood linking the model output to
the measurement data: ``\\pi(m \\mid y, \\sigma)``.

For `distribution = Normal`, the measurement is assumed to be normally distributed with
``m \\sim \\mathcal{N}(y, \\sigma^2)``. The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}}\\mathrm{exp}\\bigg( -\\frac{(m - y)^2}{2\\sigma^2} \\bigg)
```

If ``\\sigma = 1``, this likelihood reduces to the least-squares objective function.

For `distribution = Laplace`, the measurement is assumed to be Laplace distributed with
``m \\sim \\mathcal{L}(y, \\sigma)``. The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{2\\sigma}\\mathrm{exp}\\bigg( -\\frac{|m - y|}{\\sigma} \\bigg)
```

For `distribution = LogNormal`, the log of the measurement is assumed to be Normal
distributed with ``\\mathrm{log}(m) \\sim \\mathcal{N}(\\mathrm{log}(y), \\sigma^2)``
(requires ``m > 0`` and ``y > 0``). The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}\\, m}\\mathrm{exp}\\bigg( -\\frac{\\big(\\mathrm{log}(m) - \\mathrm{log}(y)\\big)^2}{2\\sigma^2} \\bigg)
```

For `distribution = Log2Normal|Log10Normal`, similar to the `LogNormal`, ``\\log_2(m)`` and
``\\log_{10}(m)`` are assumed to be normally distributed.

For `distribution = LogLaplace`, the log of the measurement is assumed to be Laplace
distributed with ``\\mathrm{log}(m) \\sim \\mathcal{L}(\\mathrm{log}(y), \\sigma)``
(requires ``m > 0`` and ``y > 0``). The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{2\\sigma\\, m}\\mathrm{exp}\\bigg( -\\frac{\\big|\\mathrm{log}(m) - \\mathrm{log}(y)\\big|}{\\sigma} \\bigg)
```

For numerical stability, PEtab.jl works with the log-likelihood in practice.
"""
struct PEtabObservable
    observable_id::String
    observable_formula::String
    noise_formula::String
    distribution
end
function PEtabObservable(observable_id::UserFormula, observable_formula::UserFormula, noise_formula::Union{UserFormula, Real};
                         distribution = Distributions.Normal)
    supported_dist = [getfield.(values(NOISE_DISTRIBUTIONS), :dist)]
    if !(distribution in supported_dist[1])
        throw(PEtabFormatError("Unsupported noise distribution: $(distribution). Supported \
            distributions are $(string.(supported_dist)...)"))
    end
    return PEtabObservable(string(observable_id), string(observable_formula), string(noise_formula), distribution)
end

"""
    PEtabCondition(condition_id, assignments::Pair...; t0 = 0.0)

Simulation condition that overrides model entities according to `assignments` under
`condition_id`.

Used to set initial values and/or model parameters for different experimental conditions.
For examples, see the online package documentation.

# Arguments
- `condition_id::Union{String, Symbol}`: Simulation condition identifier. Measurements
    are linked to this condition via the column `simulation_id` in the measurement table.
- `assignments`: One or more assignments of the form `target_id => target_value`.
    - `target_id`: Entity to assign (`Symbol`, `String`, or Symbolics `Num`). Can be a model
      state id or model parameter id for a parameter that is not estimated.
    - `target_value`: Value/expression assigned to `target_id`. A `String`, `Real`, or a
      Symbolics expression (`Num`) which can use standard Julia functions (e.g. exp, log,
      sin, cos). Any variables referenced must be model parameters or `PEtabParameter`s
      (model state variables are not allowed).

# Keyword Arguments
- `t0 = 0.0`: Simulation start time for the condition.
"""
struct PEtabCondition
    condition_id::String
    target_ids::Vector{String}
    target_values
    t0::Float64
end
function PEtabCondition(condition_id::Union{Symbol, AbstractString}, assignments::Pair...;
                        t0::Real = 0.0)
    condition_id = string(condition_id)
    if isempty(assignments)
        return PEtabCondition(condition_id, String[], String[], t0)
    end

    target_ids = first.(assignments)
    for (i, target_id) in pairs(target_ids)
        _check_target_id(target_id, i, condition_id)
    end
    target_ids = collect(string.(target_ids))

    target_values = last.(assignments)
    return PEtabCondition(condition_id, target_ids, target_values, t0)
end

"""
    PEtabEvent(condition, assignments::Pair...; condition_ids = [:all])

Model event triggered when `condition` transitions from `false` to `true`, applying the
updates in `assignments`.

For examples, see the online package documentation.

# Arguments
- `condition`: Boolean expression that triggers the event on a `false` → `true` transition.
  For example, `t == 3.0` triggers at `t = 3.0`; `S > 2.0` triggers when species `S` crosses
  `2.0` from below.
- `assignments`: One or more updates of the form `target_id => target_value`.
  - `target_id`: Entity to update (`Symbol`, `String`, or Symbolics `Num`). May be a model
    state id or model parameter id.
  - `target_value`: Value/expression assigned to `target_id` (`Real`, `String`, or Symbolics
    `Num`). May use standard Julia functions (e.g. exp, log, sin, cos) and reference
    model states/parameters.

# Keyword Arguments
- `condition_ids`: Simulation condition identifiers (as declared by `PEtabCondition`) for
    which the event applies. If `[:all]` (default), the event applies to all conditions.

# Event evaluation order

`target_value` expressions are evaluated at the `condition` trigger point using pre-event
model values, meaning all assignments are applied simultaneously (updates do not see each
other’s new values). If a time-triggered event fires at the same time as a measurement, the
model observable is evaluated **after** applying the event.
"""
struct PEtabEvent
    condition::String
    target_ids::Vector{String}
    target_values::Vector{String}
    trigger_time::Float64
    condition_ids::Vector{Symbol}
end
function PEtabEvent(condition::Union{UserFormula, Real}, assignments::Pair...; trigger_time::Real = NaN, condition_ids::Union{Vector{String}, Vector{Symbol}} = Symbol[])
    if isempty(assignments)
        throw(PEtabFormatError("For a PEtabEvent, at least one assignment pair \
            (target_id => target_value) must be provided."))
    end

    target_ids = first.(assignments)
    for (i, target_id) in pairs(target_ids)
        _check_target_id(target_id, i, nothing)
    end
    target_ids = collect(string.(target_ids))

    target_values = last.(assignments)
    for (i, target_value) in pairs(target_values)
        _check_target_value(target_value, i, nothing)
    end
    target_values = collect(string.(target_values))

    return PEtabEvent(string(condition), target_ids, target_values, trigger_time, Symbol.(condition_ids))
end

struct PEtabMLParameter{T <: AbstractFloat}
    ml_id::Symbol
    estimate::Bool
    value::Union{Nothing, ComponentVector{T}}
    prior::Nothing
end
function PEtabMLParameter(ml_id, estimate, value)
    return PEtabMLParameter(ml_id, estimate, value, nothing)
end

mutable struct MLModel{T1 <: Any, T2 <: Union{Vector{Symbol}, Vector{Vector{Symbol}}}}
    const model::T1
    st::NamedTuple
    const ps::ComponentVector{Float64}
    const static::Bool
    const dirdata::String
    const inputs::T2
    const outputs::Vector{Symbol}
    const array_inputs::Dict{Symbol, Array{<:Real}}
end

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
    petab_tables::PEtabTables
    callbacks::Dict{Symbol, SciMLBase.DECallback}
    defined_in_julia::Bool
    petab_events::Vector{PEtabEvent}
    sys_observables::Dict{Symbol, Function}
    ml_models::Dict{Symbol, <:MLModel}
end
