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
    PEtabObservable(observable_id, observable_formula, noise_formula; distribution = Normal)

Observation model linking model output (`observable_formula`) to measurement data via a
likelihood defined by `distribution` with noise/scale given by `noise_formula`.

For any supported likelihood `distribution`, `observable_formula` is the distribution
median, i.e., measurements are assumed equally likely to lie above or below the model output.

## Arguments

- `observable_id`: Observable identifier (`String` or `Symbol`). Used to link rows in the
  measurement table (column `obs_id`) to this observable.
- `observable_formula`: Observable expression. Two supported forms:
    - `Model-observable`: A `Symbol` identifier matching an observable defined in a Catalyst
      `ReactionSystem` `@observables` block, or a non-differential variable defined in a
      ModelingToolkit `ODESystem` `@variables` block.
    - `expression`: A `String`, `:Symbol`, `Real`, or a Symbolics expression (`Num`). Can
      include  standard Julia functions (e.g. `exp`, `log`, `sin`, `cos`). Variables may
      reference model species, model parameters, or `PEtabParameter`s. Can include
      time-point-specific parameters (see documentation).
- `noise_formula`: Noise/scale expression (String or Symbolics equation), same rules as
    `observable_formula`. May include time-point-specific parameters (see documentation).

## Keyword Arguments

- `distribution`: Measurement noise distribution. Valid options are `Normal` (default),
    `Laplace`, `LogNormal`, `Log2Normal`, `Log10Normal` and `LogLaplace`. See below for
    mathematical definition.

## Mathematical description

For a measurement `m`, model output `y = observable_formula`, and a noise parameter
`σ = noise_formula`, `PEtabObservable` defines the likelihood linking the model output to
the measurement data: ``\\pi(m \\mid y, \\sigma)``.

For `distribution = Normal`, the measurement is assumed to be normally distributed with
``m \\sim \\mathcal{N}(y, \\sigma^2)``. The likelihood formula is:

```math
\\pi(m \\mid y, \\sigma) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}}\\mathrm{exp}\\bigg( -\\frac{(m - y)^2}{2\\sigma^2} \\bigg)
```

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

For `distribution = Log2Normal` or `distribution = Log10Normal` similar to the `LogNormal`,
`log2(m)` and `log10(m)` is assumed to be normally distributed.

For `distribution = LogLaplace`, the log of the measurement is assumed to be Laplace
distributed with `\\mathrm{log}(m) \\sim \\mathcal{L}(\\mathrm{log}(y), \\sigma)`
(requires `m > 0` and `y > 0`). The likelihood formula is:

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

Used to set control parameters for different experimental conditions. For example, see
the online documentation.

# Arguments
- `condition_id::Union{String,Symbol}`: Simulation condition identifier. Measurement rows
    are linked to this condition via the measurement table column `simulation_id`.
- `assignments`: One or more assignments of the form `target_id => target_value`.
    - `target_id`: Entity id to override (`Num`, `String` or `Symbol`). Can be a model
      species id or a model parameter id for a parameter that is not estimated.
    - `target_value`: Value/expression assigned to `target_id`. A `String`, `Real`, or a
      Symbolics expression (`Num`) which can use standard Julia functions (e.g. `exp`, `log`,
      `sin`, `cos`). Any variables referenced must be model parameters or `PEtabParameter`s
      (species variables are not allowed).

# Keyword Arguments
- `t0`: Simulation start time for the condition (default: `0.0`).
"""
struct PEtabCondition
    condition_id::String
    target_ids::Vector{String}
    target_values::Vector{String}
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
    for (i, target_value) in pairs(target_values)
        _check_target_value(target_value, i, condition_id)
    end
    target_values = collect(string.(target_values))

    return PEtabCondition(condition_id, target_ids, target_values, t0)
end

"""
    PEtabEvent(condition, assignments::Pair...; condition_ids = [:all])

Model event triggered when `condition` transitions from `false` to `true`, applying the
updates in `assignments`.

For example usage, see the online package documentation.

# Arguments
- `condition`: Boolean expression that triggers the event on a `false` → `true` transition.
  Examples: `t == 3.0` triggers when simulation time `t` equals parameter `3.0`; `S > 2.0`
  triggers when species `S` crosses `2.0` from below.
- `assignments`: One or more assignments of the form `targe_id => target_value`.
    - `target_id`: Entity id to set (`Num`, `String` or `Symbol`). Can be a model
      specie id or a model parameter id.
    - `target_value`: Value/expression assigned to `target_id`. A `String`, `Real`, or a
      Symbolics expression (`Num`) which can use standard Julia functions (e.g. `exp`, `log`,
      `sin`, `cos`). Any variables referenced must be model species or model parameters.

# Keyword Arguments
- `condition_ids`: Simulation condition identifiers (`String/Symbol`) as declared by any
    `PEtabCondition`s. If [:all] (default), the event is applied to all conditions.
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
