struct SimulationInfo
    conditionids::Dict{Symbol, Vector{Symbol}}
    has_pre_equilibration::Bool
    tstarts::Dict{Symbol, Float64}
    tmaxs::Dict{Symbol, Float64}
    tsaves::Dict{Symbol, Vector{Float64}}
    imeasurements::Dict{Symbol, Vector{Int64}}
    imeasurements_t::Dict{Symbol, Vector{Vector{Int64}}}
    imeasurements_t_sol::Vector{Int64}
    smatrixindices::Dict{Symbol, UnitRange{Int64}}
    odesols::Dict{Symbol, ODESolution}
    odesols_derivatives::Dict{Symbol, ODESolution}
    odesols_preeq::Dict{Symbol, Union{ODESolution, SciMLBase.NonlinearSolution}}
    could_solve::Vector{Bool}
    callbacks::Dict{Symbol, SciMLBase.DECallback}
    tracked_callbacks::Dict{Symbol, SciMLBase.DECallback}
    sensealg::Any
end

struct ObservableNoiseMap
    estimate::Vector{Bool}
    xindices::Vector{Int32}
    constant_values::Vector{Float64}
    nparameters::Int32
    single_constant::Bool
end

struct ConditionMap
    isys_condition::Vector{Int32}
    ix_condition::Vector{Int32}
    target_value_functions::Vector{Function}
    ix_all_conditions::Vector{Int32}
    isys_all_conditions::Vector{Int32}
    jac::Matrix{Float64}
end
function (condition_map::ConditionMap)(p::AbstractVector, xdynamic::AbstractVector)::Nothing
    @views p[condition_map.ix_all_conditions] .= xdynamic[condition_map.isys_all_conditions]
    for (i, target_value_function) in pairs(condition_map.target_value_functions)
        p[condition_map.isys_condition[i]] = target_value_function(xdynamic)
    end
    return nothing
end

struct ParameterIndices
    xindices::Dict{Symbol, Vector{Int32}}
    xids::Dict{Symbol, Vector{Symbol}}
    xindices_notsys::Dict{Symbol, Vector{Int32}}
    xscale::Dict{Symbol, Symbol}
    xobservable_maps::Vector{ObservableNoiseMap}
    xnoise_maps::Vector{ObservableNoiseMap}
    condition_maps::Dict{Symbol, ConditionMap}
end

struct Priors
    logpdf::Dict{Symbol, Function}
    distribution::Dict{Symbol, Distribution{Univariate, Continuous}}
    initialisation_distribution::Dict{Symbol, Distribution{Univariate, Continuous}}
    prior_on_parameter_scale::Dict{<:Symbol, <:Bool}
    has_priors::Bool
    skip::Vector{Symbol}
end
function Priors()
    # In case the models does not have priors
    return Priors(Dict{Symbol, Function}(),
                  Dict{Symbol, Distribution{Univariate, Continuous}}(),
                  Dict{Symbol, Distribution{Univariate, Continuous}}(),
                  Dict{Symbol, Bool}(), false, Symbol[])
end

struct PEtabParameters
    nominal_value::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    parameter_id::Vector{Symbol}
    parameter_scale::Vector{Symbol}
    estimate::Vector{Bool}
    nparameters_estimate::Int64
end

struct PEtabODEProblemCache{T1 <: Vector{<:AbstractFloat},
                            T2 <: DiffCache,
                            T3 <: Vector{<:AbstractFloat},
                            T4 <: Matrix{<:AbstractFloat},
                            T5 <: Dict,
                            T6 <: Dict}
    xdynamic::T1
    xnoise::T1
    xobservable::T1
    xnondynamic::T1
    xdynamic_ps::T2
    xnoise_ps::T2
    xobservable_ps::T2
    xnondynamic_ps::T2
    xdynamic_grad::T1
    xnotode_grad::T1
    jacobian_gn::T4
    residuals_gn::T1
    forward_eqs_grad::T1
    adjoint_grad::T1
    St0::T4
    ∂h∂u::T3
    ∂σ∂u::T3
    ∂h∂p::T3
    ∂σ∂p::T3
    ∂G∂p::T3
    ∂G∂p_::T3
    ∂G∂u::T3
    dp::T1
    du::T1
    p::T3
    u::T3
    S::T4
    odesols::T4
    pode::T5
    u0ode::T6
    xdynamic_input_order::Vector{Int64}
    xdynamic_output_order::Vector{Int64}
    nxdynamic::Vector{Int64}
end

struct PEtabMeasurements{T <: Vector{<:Union{<:String, <:AbstractFloat}}}
    measurement::Vector{Float64}
    measurement_transformed::Vector{Float64}
    simulated_values::Vector{Float64}
    chi2_values::Vector{Float64}
    residuals::Vector{Float64}
    measurement_transforms::Vector{Symbol}
    time::Vector{Float64}
    observable_id::Vector{Symbol}
    pre_equilibration_condition_id::Vector{Symbol}
    simulation_condition_id::Vector{Symbol}
    noise_parameters::T
    observable_parameters::Vector{String}
    simulation_start_time::Vector{Float64}
end

struct ModelInfo
    petab_measurements::PEtabMeasurements
    petab_parameters::PEtabParameters
    xindices::ParameterIndices
    simulation_info::SimulationInfo
    priors::Priors
    model::PEtabModel
    nstates::Int32
end
function ModelInfo(model::PEtabModel, sensealg, custom_values)::ModelInfo
    @unpack petab_tables, callbacks, petab_events = model
    petab_measurements = PEtabMeasurements(petab_tables[:measurements], petab_tables[:observables])
    petab_parameters = PEtabParameters(petab_tables[:parameters], custom_values = custom_values)
    xindices = ParameterIndices(petab_parameters, petab_measurements, model)
    simulation_info = SimulationInfo(callbacks, petab_measurements, petab_events; sensealg = sensealg)
    priors = Priors(xindices, petab_tables[:parameters])
    nstates = Int32(length(unknowns(model.sys_mutated)))
    return ModelInfo(petab_measurements, petab_parameters, xindices, simulation_info,
                     priors, model, nstates)
end

"""
    ODESolver(solver, kwargs..)

ODE-solver options (`solver`, `tolerances`, etc.) used for solving the ODE model in a
`PEtabODEProblem`.

Any `solver` from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and
[Sundials.jl](https://github.com/SciML/Sundials.jl) is supported. For solver recommendations
and default options used when an `ODESolver` is not provided when creating a
`PEtabODEProblem`, see the documentation.

More information on the available options and solvers can also be found in the
documentation for
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).

# Keyword Arguments
- `abstol=1e-8`: Absolute tolerance when solving the ODE model. It is not recommended to
    increase above 1e-6 for gradients in order to obtain accurate gradients.
- `reltol=1e-8`: Absolute tolerance when solving the ODE model. It is not recommended to
    increase above 1e-6 for gradients in order to obtain accurate gradients.
- `dtmin=nothing`: Minimum acceptable step size when solving the ODE model.
- `maxiters=10000`: Maximum number of iterations when solving the ODE model. Increasing
    above the default value can cause parameter estimation to take substantial longer time.
- `verbose::Bool=true`: Whether or not warnings are displayed if the solver exits early.
    `true` is recommended to detect if a suboptimal choice of `solver`.
- `adj_solver=solver`: Solver to use when solving the adjoint ODE. Only applicable if
    `gradient_method=:Adjoint` when creating the `PEtabODEProblem`. Defaults to `solver`.
- `abstol_adj=abstol`: Absolute tolerance when solving the adjoint ODE model. Only
    applicable if `gradient_method=:Adjoint` when creating the `PEtabODEProblem`. Defaults
    to `abstol`.
- `reltol_adj=abstol`: Relative tolerance when solving the adjoint ODE model. Only
    applicable if `gradient_method=:Adjoint` when creating the `PEtabODEProblem`. Defaults
    to `reltol`.
"""
mutable struct ODESolver
    solver::SciMLAlgorithm
    solver_adj::SciMLAlgorithm
    abstol::Float64
    reltol::Float64
    abstol_adj::Float64
    reltol_adj::Float64
    force_dtmin::Bool
    dtmin::Union{Float64, Nothing}
    maxiters::Int64
    verbose::Bool
end
function ODESolver(solver::SciMLAlgorithm;
                   abstol::Float64 = 1e-8,
                   reltol::Float64 = 1e-8,
                   solver_adj::Union{Nothing, SciMLAlgorithm} = nothing,
                   abstol_adj::Union{Nothing, Float64} = nothing,
                   reltol_adj::Union{Nothing, Float64} = nothing,
                   force_dtmin::Bool = false,
                   dtmin::Union{Float64, Nothing} = nothing,
                   maxiters::Int64 = Int64(1e4),
                   verbose::Bool = true)
    _solver_adj = isnothing(solver_adj) ? solver : solver_adj
    _abstol_adj = isnothing(abstol_adj) ? abstol : abstol_adj
    _reltol_adj = isnothing(reltol_adj) ? reltol : reltol_adj

    return ODESolver(solver, _solver_adj, abstol, reltol, _abstol_adj, _reltol_adj,
                     force_dtmin, dtmin, maxiters, verbose)
end

"""
    SteadyStateSolver(method::Symbol; kwargs...)

Steady-state solver options (`method`, `tolerances`, etc.) for computing the steady state,
where the ODE right-hand side `f` fulfills `du = f(u, p, t) ≈ 0`.

The steady state can be computed in two ways. If `method=:Simulate`, by simulating
the ODE model until `du = f(u, p, t) ≈ 0` using ODE solver options defined in the provied
`ODESolver` to the `PEtabODEProblem. This approach is **strongly** recommended. If
`method=:Rootfinding`, by directly finding the root of the RHS `f(u, p, t) = 0` using
any algorithm from NonlinearSolve.jl. The root-finding approach is far less reliable than
the simulation approach (see description below).

## Keyword Arguments
- `termination_check = :wrms`: Approach to check if the model has reached steady-state for
    `method = :Simulate`. Two approaches are supported:
  - `:wrms`: Weighted root-mean-square. Terminate when:
    ``\\sqrt{\\frac{1}{N} \\sum_i^N \\Big( \\frac{du_i}{reltol * u_i + abstol}\\Big)^2} < 1``
  - `:Newton`: Terminate if the step for Newton's method `Δu` is sufficiently small:
    ``\\sqrt{\\frac{1}{N} \\sum_i^N \\Big(\\frac{\\Delta u_i}{reltol * u_i + abstol}\\Big)^2} < 1`` \
    The `:Newton` approach requires that the Newton step `Δu` can be computed, which is
    only possible if the Jacobian of the RHS of the ODE model is invertible. If this is not
    the case, a pseudo-inverse is used if `pseudoinverse = true`, else `wrms` is used. The
    `:Newton` termination should perform better than `:wrms` as it accounts for the scale
    of the ODEs, but until benchmarks confirm this, we recommend `:wrms`.
- `pseudoinverse = true`: Whether to use a pseudo-inverse if inverting the Jacobian fails
    for `termination_check = :Newton`.
- `rootfinding_alg = nothing`: Root-finding algorithm from NonlinearSolve.jl to use if
    `method = :Rootfinding`. If left empty, the default NonlinearSolve.jl algorithm is used.
- `abstol`: Absolute tolerance for the NonlinearSolve.jl root-finding problem or the
    `termination_check` formula. Defaults to `100 * abstol_ode`, where `abstol_ode` is the
    tolerance for the `ODESolver` in the `PEtabODEProblem`.
- `reltol`: Relative tolerance for the NonlinearSolve.jl root-finding problem or the
    `termination_check` formula. Defaults to `100 * reltol_ode`, where `reltol_ode` is the
    tolerance for the `ODESolver` in the `PEtabODEProblem`.
- `maxiters`: Maximum number of iterations to use if `method = :Rootfinding`.

## Description

For an ODE model of the form:

```math
\\frac{\\mathrm{d}u}{\\mathrm{d}t} = du = f(u, p, t)
```

The steady state is defined by:

```math
f(u, p, t) = 0.0
```

The steady state can be computed in two ways: either by simulating the ODE model from a set
of initial values `u₀` until `du ≈ 0`, or by using a root-finding algorithm such as
[Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) to directly solve
`f(u, p, t) = 0.0`. While the root-finding approach is computationally more efficient
(since it does not require solving the entire ODE), it is far less reliable and can
converge to the wrong root. For example, in mass-action models with positive initial values,
the feasible root should be positive in `u`. This is generally fulfilled when computing the
steady state via simulation (a negative root typically only occurs if the ODE solver fails).
However, with root-finding, there is no such guarantee, as any root that satisfies
`f(u, p, t) = 0.0` may be returned. Consistent with this, benchmarks have shown that
simulation is the most reliable method [1].

Another alternative is to solve for the steady state  symbolically. If feasible, this is
the most computationally efficient approach [1].

1. Fiedler et al, BMC system biology, pp 1-19 (2016)
"""
struct SteadyStateSolver{T1 <:
                         Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm},
                         T2 <: Union{Nothing, AbstractFloat},
                         T3 <: Union{Nothing, NonlinearProblem},
                         CA <: Union{Nothing, SciMLBase.DECallback},
                         T4 <: Union{Nothing, Integer}}
    method::Symbol
    rootfinding_alg::T1
    termination_check::Symbol
    abstol::T2
    reltol::T2
    maxiters::T4
    callback_ss::CA
    nprob::T3
    pseudoinverse::Bool
    tmin_simulate::Vector{Float64}
end
function SteadyStateSolver(method::Symbol; termination_check::Symbol = :wrms,
                           rootfinding_alg::NonlinearAlg = nothing, abstol = nothing,
                           reltol = nothing, pseudoinverse::Bool = false,
                           maxiters::Union{Nothing, Int64} = nothing)::SteadyStateSolver
    if !(method in [:Rootfinding, :Simulate])
        throw(PEtabInputError("Allowed methods for computing steady state are :Rootfinding \
                               :Simulate not $method"))
    end
    if method === :Simulate
        return SteadyStateSolver(termination_check, abstol, reltol, maxiters, pseudoinverse)
    else
        return SteadyStateSolver(rootfinding_alg, abstol, reltol, maxiters)
    end
end
function SteadyStateSolver(termination_check::Symbol, abstol, reltol, maxiters,
                           pseudoinverse::Bool)::SteadyStateSolver
    if !(termination_check in [:Newton, :wrms])
        throw(PEtabInputError("When steady states are computed via simulations \
                               allowed termination methods are :Newton or :wrms not \
                               $check_termination"))
    end
    return SteadyStateSolver(:Simulate, nothing, termination_check, abstol, reltol,
                             maxiters, nothing, nothing, pseudoinverse, [Inf])
end
function SteadyStateSolver(alg::NonlinearAlg, abstol, reltol, maxiters)::SteadyStateSolver
    _alg = isnothing(alg) ? NonlinearSolve.TrustRegion() : alg
    return SteadyStateSolver(:Rootfinding, _alg, :nothing, abstol, reltol, maxiters,
                             nothing, nothing, false, [Inf])
end

struct PEtabODEProblemInfo{S1 <: ODESolver, S2 <: ODESolver, C <: PEtabODEProblemCache}
    odeproblem::ODEProblem
    odeproblem_gradient::ODEProblem
    solver::S1
    solver_gradient::S2
    ss_solver::SteadyStateSolver
    ss_solver_gradient::SteadyStateSolver
    gradient_method::Symbol
    hessian_method::Symbol
    FIM_method::Symbol
    reuse_sensitivities::Bool
    sparse_jacobian::Bool
    sensealg::Any
    sensealg_ss::Any
    cache::C
    split_over_conditions::Bool
    chunksize::Int64
end

"""
    PEtabODEProblem(model::PEtabModel; kwargs...)

From `model` create a `PEtabODEProblem` for parameter estimation/inference.

If no options (`kwargs`) are provided, default settings are used for the `ODESolver`,
gradient method, and Hessian method. For more information on the default options, see the
documentation.

See also [`ODESolver`](@ref), [`SteadyStateSolver`](@ref), and [`PEtabModel`](@ref).

# Keyword Arguments

- `odesolver::ODESolver`: ODE solver options for computing the likelihood (objective
    function).
- `odesolver_gradient::ODESolver`: ODE solver options for computing the gradient. Defaults
    to `odesolver` if not provided.
- `ss_solver::SteadyStateSolver`: Steady-state solver options for computing the likelihood.
    Only applicable for models with steady-state simulations.
- `ss_solver_gradient::SteadyStateSolver`: Steady-state solver options for computing the
    gradient. Defaults to `ss_solver` if not provided.
- `gradient_method::Symbol`: Method for computing the gradient. Available options and
    defaults are listed in the documentation.
- `hessian_method::Symbol`: Method for computing the Hessian. As with the gradient,
    available options and defaults can be found in the documentation.
- `FIM_method=nothing`: Method for computing the empirical Fisher Information Matrix (FIM).
    Accepts the same options as `hessian_method`.
- `sparse_jacobian=false`: Whether to use a sparse Jacobian when solving the ODE model.
    This can greatly improve performance for large models.
- `sensealg`: Sensitivity algorithm for gradient computations. Available and default
    options depend on `gradient_method`. See the documentation for details.
- `chunksize=nothing`: Chunk size for ForwardDiff.jl when using forward-mode automatic
    differentiation for gradients and Hessians. If not provided, a default value is used.
    Tuning `chunksize` can improve performance but is non-trivial.
- `split_over_conditions::Bool=false`: Whether to split ForwardDiff.jl calls across
    simulation conditions when computing the gradient and/or Hessian. This improves
    performance for models with many condition-specific parameters, otherwise it increases
    runtime.
- `reuse_sensitivities::Bool=false`: Whether to reuse forward sensitivities from the
    gradient computation for the Gauss-Newton Hessian approximation. Only applies when
    `hessian_method=:GaussNewton` and `gradient_method=:ForwardEquations`. This can greatly
    improve  performance when the optimization algorithm computes both the gradient and
    Hessian simultaneously.
- `verbose::Bool = false`: Whether to print progress while building the `PEtabODEProblem`.

## Returns

A `PEtabODEProblem`, where the key fields are:

- `nllh`: Compute the negative log-likelihood function for an input vector `x`;
    `nllh(x)`. For this function and the ones below listed below, the input `x` can be
    either a `Vector` or a `ComponentArray`.
- `grad!`: Compute the in-place gradient of the nllh; `grad!(g, x)`.
- `grad`: Compute the out-of-place gradient of the nllh; `g = grad(x)`.
- `hess!`: Compute the in-place Hessian of the nllh; `hess!(H, x)`.
- `hess`: Compute the out-of-place Hessian of the nllh; `H = hess(x)`.
- `FIM`: Compute the out-of-place empirical Fisher Information Matrix (FIM) of the nllh;
    `F = FIM(x)`.
- `chi2`: Compute the chi-squared test statistic for the nllh (see mathematical definition
    below); `χ² = chi2(x)`.
- `residuals`: Computes the residuals between the measurement data and model output (see
    the mathematical definition below); `r = residuals(x)`.
- `simulated_values`: Computes the corresponding model values for each measurement in the
    measurements table, in the same order as the measurements appear.
- `lower_bounds`: Lower parameter bounds for parameter estimation, as specified when
    creating the `model`.
- `upper_bounds`: Upper parameter bounds for parameter estimation, as specified when
    creating the `model`.

## Description

Following the [PEtab standard](https://petab.readthedocs.io/en/latest/), the objective
function to be used for parameter estimation created by the `PEtabODEProblem` is a
likelihood function, or, if priors are provided, a posterior function. The characteristics
of the objective is defined in the `PEtabModel`. In practice, for numerical stability,
a `PEtabODEProblem` works with the negative log-likelihood:

```math
-\\ell(\\mathbf{x}) =  - \\sum_{i = 1}^N \\ell_i(\\mathbf{x}),
```

where ``\\ell_i`` is the likelihood for each measurement ``i``. In addition, to accommodate
numerical optimization packages, the `PEtabODEProblem` provides efficient functions for
computing the gradient ``-\\nabla \\ell(\\mathbf{x})`` and the second derivative Hessian
matrix ``-\\nabla^2 \\ell(\\mathbf{x})``, using the specified or default `gradient_method`
and `hessian_method`.

In addition to ``-\\ell`` and its derivatives, the `PEtabODEProblem` provides functions for
diagnostics and model selection. The χ² (chi-squared) value can be used for model
selection [1], and it is computed as the sum of the χ² values for each measurement. For a
measurement `y`, an observable `h = obs_formula`, and a standard deviation
`σ = noise_formula`, the χ² is computed as:

```math
\\chi^2 = \\frac{(y - h)^2}{\\sigma^2}
```

The residuals ``r`` can be used to assess the measurement error model and are computed as:

```math
r = \\frac{(y - h)}{\\sigma}
```

Lastly, the empirical Fisher Information Matrix (FIM) can be used for identifiability
analysis [2]. It should ideally be computed with an exact Hessian method. The inverse of
the FIM provides a lower bound on the covariance matrix. While the FIM can be useful, the
profile-likelihood approach generally yields better results for identifiability analysis [2].

1. Cedersund et al, The FEBS journal, pp 903-922 (2009).
2. Raue et al, Bioinformatics, pp 1923-1929 (2009).
"""
struct PEtabODEProblem{F1 <: Function, F2 <: Function, F3 <: Function}
    nllh::F1
    chi2::Any
    grad!::F2
    grad::Function
    hess!::F3
    hess::Function
    FIM!::Any
    FIM::Any
    nllh_grad::Function
    prior::Function
    grad_prior::Function
    hess_prior::Function
    simulated_values::Any
    residuals::Any
    probinfo::PEtabODEProblemInfo
    model_info::ModelInfo
    nparameters_estimate::Int64
    xnames::Vector{Symbol}
    xnominal::ComponentArray{Float64}
    xnominal_transformed::ComponentArray{Float64}
    lower_bounds::ComponentArray{Float64}
    upper_bounds::ComponentArray{Float64}
end

struct PEtabFileError <: Exception
    var::String
end

struct PEtabFormatError <: Exception
    var::String
end

struct PEtabInputError <: Exception
    var::String
end
