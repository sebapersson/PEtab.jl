struct SimulationInfo
    conditionids::Dict{Symbol, Vector{Symbol}}
    has_pre_equilibration::Bool
    tstarts::Dict{Symbol, Float64}
    tmaxs::Dict{Symbol, Float64}
    tsaves::Dict{Symbol, Vector{Float64}}
    tsaves_no_cbs::Dict{Symbol, Vector{Float64}}
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

struct MLModelPreODEMap{T1 <: Vector{<:Array{<:AbstractFloat}}}
    ninput_arguments::Int64
    constant_inputs::T1
    iconstant_inputs::Vector{Vector{Int32}}
    ixdynamic_mech_inputs::Vector{Vector{Int32}}
    ixdynamic_inputs::Vector{Vector{Int32}}
    ninputs::Vector{Int64}
    nxdynamic_inputs::Vector{Int64}
    noutputs::Int64
    ix_nn_outputs::Vector{Int32}
    ix_nn_outputs_grad::Vector{Int32}
    ioutput_sys::Vector{Int32}
    file_input::Vector{Bool}
end

struct ParameterIndices
    xids::Dict{Symbol, Vector{Symbol}}
    indices_est::Dict{Symbol, Vector{Int32}}
    indices_dynamic::Dict{Symbol, Vector{Int32}}
    indices_not_system::Dict{Symbol, Vector{Int32}}
    xscale::Dict{Symbol, Symbol}
    xobservable_maps::Vector{ObservableNoiseMap}
    xnoise_maps::Vector{ObservableNoiseMap}
    condition_maps::Dict{Symbol, ConditionMap}
    maps_ml_pre_simulate::Dict{Symbol, Dict{Symbol, MLModelPreODEMap}}
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

struct PEtabMLParameters{T <: Vector{<:Union{String, <:Float64}}}
    nominal_value::T
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    parameter_id::Vector{Symbol}
    estimate::Vector{Bool}
    ml_id::Vector{Symbol}
    mapping_table_id::Vector{String}
    initialisation_priors::Vector{Function}
end

struct PEtabODEProblemCache{T1 <: Vector{<:AbstractFloat},
                            T2 <: DiffCache,
                            T3 <: Vector{<:AbstractFloat},
                            T4 <: Matrix{<:AbstractFloat},
                            T5 <: Dict,
                            T6 <: Dict,
                            T7 <: Dict{Symbol, <:DiffCache},
                            T8 <: Union{Vector{<:AbstractFloat}, <:ComponentVector}}
    xdynamic_mech::T2
    xnoise::T2
    xobservable::T2
    xnondynamic_mech::T2
    xdynamic_ps::T2
    xnoise_ps::T2
    xobservable_ps::T2
    xnondynamic_mech_ps::T2
    xdynamic_grad::T1
    x_not_system_grad::T1
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
    p::T8
    u::T3
    S::T4
    odesols::T4
    pode::T5
    u0ode::T6
    x_ml_models_cache::T7
    x_ml_models::Dict{Symbol, ComponentArray}
    x_ml_models_constant::Dict{Symbol, ComponentArray}
    xdynamic::T2
    grad_ml_pre_simulate_outputs::Vector{Float64}
end

struct PEtabMeasurements{T <: Vector{<:Union{<:String, <:AbstractFloat}}}
    measurements::Vector{Float64}
    measurements_transformed::Vector{Float64}
    simulated_values::Vector{Float64}
    chi2_values::Vector{Float64}
    residuals::Vector{Float64}
    noise_distributions::Vector{Symbol}
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
    petab_ml_parameters::PEtabMLParameters
    xindices::ParameterIndices
    simulation_info::SimulationInfo
    priors::Priors
    model::PEtabModel
    nstates::Int32
end
function ModelInfo(model::PEtabModel, sensealg, custom_values)::ModelInfo
    @unpack petab_tables, callbacks, petab_events = model
    petab_measurements = PEtabMeasurements(petab_tables[:measurements], petab_tables[:observables])
    petab_parameters = PEtabParameters(petab_tables[:parameters], petab_tables[:mapping], model.ml_models; custom_values = custom_values)
    petab_ml_parameters = PEtabMLParameters(petab_tables[:parameters], petab_tables[:mapping], model.ml_models)
    xindices = ParameterIndices(petab_parameters, petab_measurements, model)
    simulation_info = SimulationInfo(callbacks, petab_measurements, petab_events; sensealg = sensealg)
    priors = Priors(xindices, model)
    nstates = Int32(length(_get_state_ids(model.sys_mutated)))
    return ModelInfo(petab_measurements, petab_parameters, petab_ml_parameters, xindices, simulation_info, priors, model, nstates)
end

"""
    ODESolver(solver; kwargs...)

ODE solver configuration (solver, tolerances, etc.) used to simulate the model in a
`PEtabODEProblem`.

Any ODE solver from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) or
[Sundials.jl](https://github.com/SciML/Sundials.jl) is supported. Solver recommendations and
the default configuration (when `ODESolver` is not provided) are described in the
online documentation.

# Keyword Arguments
- `abstol = 1e-8`: Absolute tolerance for the forward ODE solve. For gradient-based
  estimation, values larger than `1e-6` are generally not recommended, as they may yield
  inaccurate gradients.
- `reltol = 1e-8`: Relative tolerance for the forward ODE solve. For gradient-based
  estimation, values larger than `1e-6` are generally not recommended, as they may yield
  inaccurate gradients.
- `dtmin = nothing`: Minimum step size for the forward ODE solve.
- `maxiters = 10_000`: Maximum number of solver iterations. Increasing this can
  substantially increase runtime during parameter estimation.
- `verbose::Bool = true`: Whether to print warnings if the solver terminates early.
  Keeping this enabled is recommended to detect problematic solver configurations.
- `adj_solver = solver`: ODE solver used for the adjoint solve. Defaults to `solver`. Only
  relevant when `gradient_method = :Adjoint`.
- `abstol_adj = abstol`: Absolute tolerance for the adjoint solve (only relevant when
  `gradient_method = :Adjoint`).
- `reltol_adj = reltol`: Relative tolerance for the adjoint solve (only relevant when
  `gradient_method = :Adjoint`).
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

Options for computing a steady state, i.e. a state `u` where the ODE right-hand side
`f(u, p, t)` satisfies `du = f(u, p, t) ≈ 0`.

Two approaches are supported:

- `method = :Simulate`: Simulate the ODE forward in time until a steady-state termination
  criterion is met. The simulation uses the `ODESolver` settings provided to the
  `PEtabODEProblem`. This is the **recommended** and most robust approach.
- `method = :Rootfinding`: Solve `f(u, p, t) = 0` directly using a root-finding method from
  NonlinearSolve.jl. This can be faster, but is generally less reliable (see below).

# Keyword Arguments
- `termination_check = :wrms`: Termination criterion used for `method = :Simulate`. Options:
  - `:wrms`: Weighted root-mean-square of `du`. Terminate when
    ``\\sqrt{\\frac{1}{N}\\sum_{i=1}^N\\left(\\frac{du_i}{\\mathrm{reltol}\\,u_i + \\mathrm{abstol}}\\right)^2} < 1``.
  - `:Newton`: Terminate when the Newton step `Δu` is sufficiently small:
    ``\\sqrt{\\frac{1}{N}\\sum_{i=1}^N\\left(\\frac{\\Delta u_i}{\\mathrm{reltol}\\,u_i + \\mathrm{abstol}}\\right)^2} < 1``.
    This requires solving a linear system involving the Jacobian of `f`. If the Jacobian is
    singular, a pseudo-inverse is used when `pseudoinverse = true`; otherwise the criterion
    falls back to `:wrms`. `:wrms` is recommended.
- `pseudoinverse = true`: Use a pseudo-inverse when the Jacobian inversion fails for
  `termination_check = :Newton`.
- `rootfinding_alg = nothing`: NonlinearSolve.jl algorithm used for `method = :Rootfinding`.
  If `nothing`, the NonlinearSolve.jl default is used.
- `abstol`: Absolute tolerance used both in the termination criterion and (for
  `method = :Rootfinding`) the NonlinearSolve.jl solve. Defaults to `100 * abstol_ode`,
  where `abstol_ode` is taken from the `ODESolver` in the `PEtabODEProblem`.
- `reltol`: Relative tolerance used both in the termination criterion and (for
  `method = :Rootfinding`) the NonlinearSolve.jl solve. Defaults to `100 * reltol_ode`,
  where `reltol_ode` is taken from the `ODESolver` in the `PEtabODEProblem`.
- `maxiters`: Maximum number of iterations for `method = :Rootfinding`.

# Mathemathical description

For an ODE model

```math
\\frac{\\mathrm{d}u}{\\mathrm{d}t} = f(u, p, t).
```

A steady state is a solution ``u^{*}`` such that

```math
f(u^*, p, t) = 0.
```

The steady state can be obtained either by simulation (`:Simulate`) or by direct
root-finding (`:Rootfinding`). Although root-finding can be more computationally efficient,
it may converge to an unintended root. For example, mass-action models with positive initial
conditions often have a physically meaningful steady state with `u ≥ 0`. Simulation tends to
preserve such invariants (except in case of solver failure), while a generic root-finder has
no such guarantees and may return a different root that still satisfies `f(u, p, t) = 0`.
In line with this, benchmarks show that `:Simulate` is more robust [1]. Lastly, if the
steady state can be solved for symbolically, it is the best approach [1].

1. Fiedler et al., *BMC Systems Biology* (2016), pp. 1–19.
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

struct MLModelPreODE{T1 <: DiffCache, T2 <: Vector{<:DiffCache}}
    forward!::Function
    tape::Any
    jac_ml_model::Matrix{Float64}
    outputs::T1
    inputs::T2
    x::T1
    computed::Vector{Bool}
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
    ml_models_pre_ode::Dict{Symbol, Dict{Symbol, MLModelPreODE}}
end

"""
    PEtabODEProblem(model::PEtabModel; kwargs...) -> PEtabODEProblem

Create a `PEtabODEProblem` from `model` for parameter estimation and Bayesian inference.

If no keyword arguments are provided, PEtab.jl chooses defaults for the `ODESolver`,
gradient method, and Hessian method (see the online documentation on default options).

# Keyword Arguments

- `odesolver::ODESolver`: ODE solver options used when evaluating the objective (negative
  log-likelihood).
- `odesolver_gradient::ODESolver`: ODE solver options used when evaluating gradients.
  Defaults to `odesolver`.
- `ss_solver::SteadyStateSolver`: Steady-state solver options used when evaluating the
  objective. Only applicable for models with steady-state (pre-equilibration) simulations.
- `ss_solver_gradient::SteadyStateSolver`: Steady-state solver options used when evaluating
  gradients. Defaults to `ss_solver`.
- `gradient_method::Symbol`: Method used to compute gradients. Available options and
  defaults are described in the online documentation.
- `hessian_method::Symbol`: Method used to compute Hessians. Available options and defaults
  are described in the online documentation.
- `FIM_method = nothing`: Method used to compute the empirical Fisher Information Matrix
  (FIM). Accepts the same options as `hessian_method`. If `nothing`, the default is used.
- `sparse_jacobian::Bool = false`: Whether to use a sparse Jacobian when solving the ODE.
  Can substantially improve performance for large models.
- `sensealg = nothing`: Sensitivity algorithm used for gradient computations. Available and
  default options depend on `gradient_method`. See the documentation for details.
- `chunksize = nothing`: Chunk size used by ForwardDiff.jl for forward-mode automatic
  differentiation (gradients and Hessians). If not provided, a default is chosen. Tuning
  `chunksize` can improve performance, but the optimal value is model-dependent.
- `split_over_conditions::Bool = false`: Whether to split ForwardDiff-based derivative
  computations across simulation conditions. Can improve performance for models with many
  condition-specific parameters, but adds overhead otherwise.
- `reuse_sensitivities::Bool = false`: Whether to reuse forward sensitivities from gradient
  computations when forming a Gauss-Newton Hessian approximation. Only applies when
  `hessian_method = :GaussNewton` and `gradient_method = :ForwardEquations`. This can
  substantially improve performance when the optimizer evaluates gradient and Hessian
  together.
- `verbose::Bool = false`: Whether to print progress while building the `PEtabODEProblem`.

# Returns

A `PEtabODEProblem`, which provides everything needed for wrapping an optmization algorithm,
key fields are:

- `nllh`: Negative log-likelihood function (or negative log-posterior if priors are
  included); `nllh(x)`.
- `grad!`: In-place gradient function; `grad!(g, x)`.
- `grad`: Out-of-place gradient function; `g = grad(x)`.
- `nllh_grad`: Function to compute `nllh` and `grad` simultanesouly:
  `nllh, g = nllh_grad(x)`. More efficient than calling `nllh` and `grad` separately since
  quantities from computing `grad` are resued for `nllh`.
- `hess!`: In-place Hessian function; `hess!(H, x)`.
- `hess`: Out-of-place Hessian function; `H = hess(x)`.
- `FIM`: Empirical Fisher Information Matrix function; `F = FIM(x)`.
- `chi2`: Chi-squared test statistic function; `χ² = chi2(x)`.
- `residuals`: Residual vector function; `r = residuals(x)`.
- `simulated_values`: Model-simulated values corresponding to each measurement row, in the
  same order as the measurement table.
- `lower_bounds`, `upper_bounds`: Parameter bounds used for optimization/inference.

Unless otherwise noted, the input `x` can be a `Vector` or a `ComponentArray` (the output
type matches the input type where applicable). `x` must be provided in the order expected
by a `PEtabODEProblem`, see [`get_x`](@ref)

See also: [`PEtabModel`](@ref), [`ODESolver`](@ref), [`SteadyStateSolver`](@ref).

# Mathemathical description

Following the [PEtab standard](https://petab.readthedocs.io/en/latest/), the objective
function for a `PEtabODEProblem` is a likelihood (or posterior, when priors are included).
For numerical stability, `PEtabODEProblem` works with the negative log-likelihood; so
`nllh` equals:

```math
-\\ell(\\mathbf{x}) = -\\sum_{i=1}^{N} \\ell_i(\\mathbf{x}),
```

where ``\\ell_i`` is the log-likelihood contribution from measurement `i```. The
`gradient_method` and `hessian_method` determine how ``-\\nabla \\ell(\\mathbf{x})`` and
``-\\nabla^2 \\ell(\\mathbf{x})`` are computed.

In addition to ``-\\ell`` and its derivatives, `PEtabODEProblem` provides diagnostics and
quantities useful for model assessment. The chi-squared value is computed (per measurement)
as [1]:

```math
\\chi^2 = \\frac{(y - h)^2}{\\sigma^2},
```

where `y` is the measurement, `h` is the observable value (`obs_formula`), and
`σ` is the noise standard deviation (`noise_formula`). Residuals are computed as

```math
r = \\frac{(y - h)}{\\sigma}.
```

The empirical Fisher Information Matrix (FIM) can be used for identifiability analysis.
When feasible, it should be computed with an exact Hessian method. The inverse of the FIM
provides a lower bound on the parameter covariance matrix. In practice, profile-likelihood
methods often provide more reliable results [2].

1. Cedersund et al., *The FEBS Journal*, pp. 903–922 (2009).
2. Raue et al., *Bioinformatics*, pp. 1923–1929 (2009).
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
