"""
    ODESolver(solver, <keyword arguments>)

ODE-solver options (solver, tolerances, etc...) to use when computing gradient/cost for a PEtabODEProblem.

More information about the available options and solvers can be found in the documentation for DifferentialEquations.jl (https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/). Recommended settings for which solver and options to use for different problems can be found below and in the documentation.

# Arguments
- `solver`: Any of the ODE solvers in DifferentialEquations.jl. For small (≤20 states) mildly stiff models, composite solvers such as `AutoVern7(Rodas5P())` perform well. For stiff small models, `Rodas5P()` performs well. For medium-sized models (≤75 states), `QNDF()`, `FBDF()`, and `CVODE_BDF()` perform well. `CVODE_BDF()` is not compatible with automatic differentiation and thus cannot be used if the gradient is computed via automatic differentiation or if the Gauss-Newton Hessian approximation is used. If the gradient is computed via adjoint sensitivity analysis, `CVODE_BDF()` is often the best choice as it is typically more reliable than `QNDF()` and `FBDF()` (fails less often).
- `abstol=1e-8`: Absolute tolerance when solving the ODE system. Not recommended to increase above 1e-6 for gradients.
- `reltol=1e-8`: Relative tolerance when solving the ODE system. Not recommended to increase above 1e-6 for gradients.
- `force_dtmin=false`: Whether or not to force `dtmin` when solving the ODE system.
- `dtmin=nothing`: Minimal acceptable step-size when solving the ODE system.
- `maxiters=10000`: Maximum number of iterations when solving the ODE system. Increasing above the default value can cause the optimization to take substantial time.
- `verbose::Bool=true`: Whether or not warnings are displayed if the solver exits early. `true` is recommended in order to detect if a suboptimal ODE solver was chosen.
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
    SteadyStateSolver(method::Symbol; termination_check::Symbol=:wrms,
                      rootfinding_alg=nothing, abstol=nothing, reltol=nothing,
                      maxiters=nothing)

Setup options for finding steady-state via either `method=:Rootfinding` or `method=:Simulate`.

For `method=:Rootfinding`, the steady-state `u*` is found by solving the problem `du = f(u, p, t) ≈ 0` with tolerances
`abstol` and `reltol` via an automatically chosen optimization algorithm (`rootfinding_alg=nothing`) or via any
provided algorithm in [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl).

For `method=:Simulate`, the steady-state `u*` is found by simulating the ODE system until `du = f(u, p, t) ≈ 0`.
Two options are available for `termination_check`:
- `:wrms` : Weighted root-mean square √(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1
- `:Newton` : If Newton-step `Δu` is sufficiently small √(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1.
        - Newton often performs better but requires an invertible Jacobian. In case it's not fulfilled, the code
          switches automatically to `:wrms`.

`maxiters` refers to either the maximum number of rootfinding steps or the maximum number of integration steps,
depending on the chosen method.
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
end
function SteadyStateSolver(method::Symbol; termination_check::Symbol = :wrms,
                           rootfinding_alg::Union{Nothing,
                                                  NonlinearSolve.AbstractNonlinearSolveAlgorithm} = nothing,
                           abstol = nothing, reltol = nothing,
                           maxiters::Union{Nothing, Int64} = nothing,
                           pseudoinverse::Bool = false)::SteadyStateSolver
    if !(method in [:Rootfinding, :Simulate])
        throw(PEtabInputError("Allowed methods for computing steady state are :Rootfinding " *
                              ":Simulate not $method"))
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
        throw(PEtabInputError("When steady states are computed via simulations " *
                              "allowed termination methods are :Newton or :wrms not " *
                              "$check_termination"))
    end
    return SteadyStateSolver(:Simulate, nothing, termination_check, abstol, reltol,
                             maxiters, nothing, nothing, pseudoinverse)
end
function SteadyStateSolver(alg::Union{Nothing,
                                      NonlinearSolve.AbstractNonlinearSolveAlgorithm},
                           abstol, reltol, maxiters)::SteadyStateSolver
    _alg = isnothing(alg) ? NonlinearSolve.TrustRegion() : alg
    return SteadyStateSolver(:Rootfinding, _alg, :nothing, abstol, reltol, maxiters,
                             nothing, nothing, false)
end

struct SimulationInfo
    conditionids::Dict{Symbol, Vector{Symbol}}
    has_pre_equilibration::Bool
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
    constant_values::Vector{Float64}
    isys_constant_values::Vector{Int32}
    ix_dynamic::Vector{Int32}
    ix_sys::Vector{Int32}
end

struct MapODEProblem
    sys_to_dynamic::Vector{Int32}
    dynamic_to_sys::Vector{Int32}
end

struct ParameterIndices
    xindices::Dict{Symbol, Vector{Int32}}
    xids::Dict{Symbol, Vector{Symbol}}
    xindices_notsys::Dict{Symbol, Vector{Int32}}
    xscale::Dict{Symbol, Symbol}
    mapxobservable::Vector{ObservableNoiseMap}
    mapxnoise::Vector{ObservableNoiseMap}
    map_odeproblem::MapODEProblem
    maps_conidition_id::Dict{Symbol, ConditionMap}
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
    nparameters_esimtate::Int64
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
    tables, cbs = model.petab_tables, model.callbacks
    petab_measurements = PEtabMeasurements(tables[:measurements], tables[:observables])
    petab_parameters = PEtabParameters(tables[:parameters], custom_values = custom_values)
    xindices = ParameterIndices(petab_parameters, petab_measurements, model)
    simulation_info = SimulationInfo(cbs, petab_measurements, sensealg = sensealg)
    priors = Priors(xindices, tables[:parameters])
    nstates = length(unknowns(model.sys_mutated)) |> Int32
    return ModelInfo(petab_measurements, petab_parameters, xindices, simulation_info,
                     priors, model, nstates)
end

"""
    PEtabODEProblem

Everything needed to setup an optimization problem (compute cost, gradient, hessian and  parameter bounds) for a PEtab model.

For constructor, see below.

!!! note
    The parameter vector θ is always assumed to be on the parameter scale specified in the PEtab parameters file. If needed, θ is transformed to the linear scale inside the function call.

## Fields
- `compute_cost`: For θ computes the negative likelihood (objective to minimize)
- `compute_chi2`: For θ compute χ2 value
- `grad!`: For θ computes in-place gradient grad!(gradient, θ)
- `grad`: For θ computes out-place gradient gradient = grad(θ)
- `compute_hessian!`: For θ computes in-place hessian-(approximation) compute_hessian!(hessian, θ)
- `compute_hessian`: For θ computes out-place hessian-(approximation) hessian = compute_hessian(θ)
- `compute_FIM!`: For θ computes the empirical Fisher-Information-Matrix (FIM) which is the Hessian of the negative-log-likelihood  compute_FIM!(FIM, θ).
- `compute_FIM`: For θ computes FIM out of place FIM = compute_FIM(θ).
- `compute_simulated_values`: For θ compute the corresponding model (simulated) values to the measurements in the same order as in the Measurements PEtab table
- `compute_residuals`: For θ compute the residuals (h_model - h_observed)^2 / σ^2 in the same order as in the Measurements PEtab table
- `gradient_method`: The method used to compute the gradient (either :ForwardDiff, :ForwardEquations, :Adjoint, or :Zygote).
- `hessian_method`: The method used to compute or approximate the Hessian (either :ForwardDiff, :BlocForwardDiff, or :GaussNewton).
- `FIM_method`: The method used to compute FIM, either :ForwardDiff (full Hessian) or :GaussNewton (only recomended for >100 parameter models)
- `nparameters_esimtate`: The number of parameters to estimate.
- `θ_names`: The names of the parameters in θ.
- `θ_nominal`: The nominal values of θ as specified in the PEtab parameters file.
- `θ_nominalT`: The nominal values of θ on the parameter scale (e.g., log) as specified in the PEtab parameters file.
- `lower_bounds`: The lower parameter bounds on the parameter scale for θ as specified in the PEtab parameters file.
- `upper_bounds`: The upper parameter bounds on the parameter scale for θ as specified in the PEtab parameters file.
- `model`: The PEtabModel used to construct the PEtabODEProblem.
- `ode_solver`: The options for the ODE solver specified when creating the PEtabODEProblem.
- `ode_solver_gradient`: The options for the ODE solver gradient specified when creating the PEtabODEProblem.

## Constructor

    PEtabODEProblem(model::PEtabModel; <keyword arguments>)

Given a `PEtabModel` creates a `PEtabODEProblem` with potential user specified options.

The keyword arguments (described below) allows to choose cost, gradient, and Hessian methods, ODE solver options,
and other tuneable options that can potentially make computations more efficient for some "edge-case" models. Please
refer to the documentation for guidance on selecting the most efficient options for different types of models. If a
keyword argument is not set, a suitable default option is chosen based on the number of model parameters.

Once created, a `PEtabODEProblem` contains everything needed to perform parameter estimtimation (see above)

!!! note
    Every problem is unique, so even though the default settings often work well they might not be optimal.

# Keyword arguments
- `ode_solver::ODESolver`: Options for the ODE solver when computing the cost, such as solver and tolerances.
- `ode_solver_gradient::ODESolver`: Options for the ODE solver when computing the gradient, such as the ODE solver options used in adjoint sensitivity analysis. Defaults to `ode_solver` if not set to nothing.
- `ss_solver::SteadyStateSolver`: Options for finding steady-state for models with pre-equilibrium. Steady-state can be found via simulation or rootfinding, which can be set using `SteadyStateSolver` (see documentation). If not set, defaults to simulation with `wrms < 1` termination.
- `ss_solver_gradient::SteadyStateSolver`: Options for finding steady-state for models with pre-equilibrium when computing gradients. Defaults to `ss_solver` value if not set.
- `cost_method::Symbol=:Standard`: Method for computing the cost (objective). Two options are available: `:Standard`, which is the most efficient, and `:Zygote`, which is less efficient but compatible with the Zygote automatic differentiation library.
- `gradient_method=nothing`: Method for computing the gradient of the objective. Four options are available:
    * `:ForwardDiff`: Compute the gradient via forward-mode automatic differentiation using ForwardDiff.jl. Most efficient for models with ≤50 parameters. The number of chunks can be optionally set using `chunksize`.
    * `:ForwardEquations`: Compute the gradient via the model sensitivities, where `sensealg` specifies how to solve for the sensitivities. Most efficient when the Hessian is approximated using the Gauss-Newton method and when the optimizer can reuse the sensitivities (`reuse_sensitivities`) from gradient computations in Hessian computations (e.g., when the optimizer always computes the gradient before the Hessian).
    * `:Adjoint`: Compute the gradient via adjoint sensitivity analysis, where `sensealg` specifies which algorithm to use. Most efficient for large models (≥75 parameters).
    * `:Zygote`: Compute the gradient via the Zygote package, where `sensealg` specifies which sensitivity algorithm to use when solving the ODE model. This is the most inefficient option and not recommended.
- `hessian_method=nothing`: method for computing the Hessian of the cost. There are three available options:
    * `:ForwardDiff`: Compute the Hessian via forward-mode automatic differentiation using ForwardDiff.jl. This is often only computationally feasible for models with ≤20 parameters but can greatly improve optimizer convergence.
    * `:BlockForwardDiff`: Compute the Hessian block approximation via forward-mode automatic differentiation using ForwardDiff.jl. The approximation consists of two block matrices: the first is the Hessian for only the dynamic parameters (parameter part of the ODE system), and the second is for the non-dynamic parameters (e.g., noise parameters). This is computationally feasible for models with ≤20 dynamic parameters and often performs better than BFGS methods.
    * `:GaussNewton`: Approximate the Hessian via the Gauss-Newton method, which often performs better than the BFGS method. If we can reuse the sensitivities from the gradient in the optimizer (see `reuse_sensitivities`), this method is best paired with `gradient_method=:ForwardEquations`.
- `FIM_method=nothing`: Method for computing the empirical Fisher-Information-Matrix (FIM), can be:
    * `:ForwardDiff` - use ForwardDiff to compute the full Hessian (FIM) matrix, default for model with ≤ 100 parameters
    * `:GaussNewton` - approximate the FIM as the Gauss-Newton Hessian approximation (only recomeded when ForwardDiff is computationally infeasible)
- `sparse_jacobian::Bool=false`: When solving the ODE du/dt=f(u, p, t), whether implicit solvers use a sparse Jacobian. Sparse Jacobian often performs best for large models (≥100 states).
- `specialize_level=SciMLBase.FullSpecialize`: Specialization level when building the ODE problem. It is not recommended to change this parameter (see https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/).
- `sensealg`: Sensitivity algorithm for gradient computations. The available options for each gradient method are:
    * `:ForwardDiff`: None (as ForwardDiff takes care of all computation steps).
    * `:ForwardEquations`: `:ForwardDiff` (uses ForwardDiff.jl and typicaly performs best) or `ForwardDiffSensitivity()` and `ForwardSensitivity()` from SciMLSensitivity.jl (https://github.com/SciML/SciMLSensitivity.jl).
    * `:Adjoint`: `InterpolatingAdjoint()` and `QuadratureAdjoint()` from SciMLSensitivity.jl.
    * `:Zygote`: All sensealg in SciMLSensitivity.jl.
- `sensealg_ss=nothing`: Sensitivity algorithm for adjoint gradient computations for steady-state simulations. The available options are `SteadyStateAdjoint()`, `InterpolatingAdjoint()`, and `QuadratureAdjoint()` from SciMLSensitivity.jl. `SteadyStateAdjoint()` is the most efficient but requires a non-singular Jacobian, and in the case of a non-singular Jacobian, the code automatically switches to `InterpolatingAdjoint()`.
- `chunksize=nothing`: Chunk-size for ForwardDiff.jl when computing the gradient and Hessian via forward-mode automatic differentiation. If nothing is provided, the default value is used. Tuning `chunksize` is non-trivial, and we plan to add automatic functionality for this.
- `split_over_conditions::Bool=false`: For gradient and Hessian via ForwardDiff.jl, whether or not to split calls to ForwardDiff across experimental (simulation) conditions. This parameter should only be set to true if the model has many parameters specific to an experimental condition; otherwise, the overhead from the calls will increase run time. See the Beer example for a case where this is needed.
- `reuse_sensitivities::Bool=false` : If set to `true`, reuse the sensitivities computed during gradient computations for the Gauss-Newton Hessian approximation. This option is only applicable when using `hessian_method=:GaussNewton` and `gradient_method=:ForwardEquations`. Note that it should only be used when the optimizer always computes the gradient before the Hessian.
- `verbose::Bool=true` : If set to `true`, print progress messages while setting up the PEtabODEProblem.
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
    nparameters_esimtate::Int64
    xnames::Vector{Symbol}
    xnominal::ComponentArray{Float64}
    xnominal_transformed::ComponentArray{Float64}
    lower_bounds::ComponentArray{Float64}
    upper_bounds::ComponentArray{Float64}
end

"""
    Fides(hessian_method::Union{Nothing, Symbol}; verbose::Bool=false)

[Fides](https://github.com/fides-dev/fides) is a Python Newton-trust region optimizer designed for box-bounded optimization problems.

It excels when the full Hessian is too computationally expensive, but a Gauss-Newton Hessian approximation can be calculated. When constructed with `Fides(verbose=true)`, it displays optimization progress during estimation.

## Hessian Methods

If `hessian_method=nothing`, the Hessian method from the `PEtabODEProblem` is used, which can be either exact or a Gauss-Newton approximation. Additionally, Fides supports the following Hessian approximation methods:

- `:BB`: Broyden's "bad" method
- `:BFGS`: Broyden-Fletcher-Goldfarb-Shanno update strategy
- `:BG`: Broyden's "good" method
- `:Broyden`: BroydenClass Update scheme
- `:SR1`: Symmetric Rank 1 update
- `:SSM`: Structured Secant Method
- `:TSSM`: Totally Structured Secant Method

For more information on each method, see the Fides [documentation](https://fides-optimizer.readthedocs.io/en/latest/generated/fides.hessian_approximation.html).


## Examples
```julia
# Fides with the Hessian method as in the PEtabProblem
fides_opt = Fides(nothing)
```
```julia
# Fides with the BFGS Hessian approximation, with progress printing
fides_opt = Fides(:BFGS; verbose=true)
```
"""
struct Fides
    hessian_method::Union{Nothing, Symbol}
    verbose::Bool
end
function Fides(hessian_method::Union{Nothing, Symbol}; verbose::Bool = false)
    verbose_arg = verbose == true ? 1 : 0
    if isnothing(hessian_method)
        return Fides(hessian_method, verbose_arg)
    end
    allowed_approximations = [:BB, :BFGS, :BG, :Broyden, :DFB, :FX, :SR1, :SSM, :TSSM]
    @assert hessian_method ∈ allowed_approximations "Hessian approximation method $hessian_method is not allowed, see documentation on Fides for allowed methods"
    return Fides(hessian_method, verbose_arg)
end

"""
    IpoptOptimiser(LBFGS::Bool)

[Ipopt](https://coin-or.github.io/Ipopt/) is an Interior-point Newton method designed for nonlinear optimization.

Ipopt can be configured to use either the Hessian method from the `PEtabODEProblem` (`LBFGS=false`) or a LBFGS scheme (`LBFGS=true`).
For setting Ipopt options, see [`IpoptOptions`](@ref).

See also [`calibrate_model`](@ref) and [`calibrate_multistart`](@ref).

## Examples
```julia
# Ipopt with the Hessian method as in the PEtabProblem
ipopt_opt = IpoptOptimiser(false)
```
```julia
# Ipopt with LBFGS Hessian approximation
ipopt_opt = IpoptOptimiser(true)
```
"""
struct IpoptOptimiser
    LBFGS::Bool
end

"""
    IpoptOptions(;print_level::Int64=0,
                 max_iter::Int64=1000,
                 tol::Float64=1e-8,
                 acceptable_tol::Float64=1e-6,
                 max_wall_time::Float64=1e20,
                 acceptable_obj_change_tol::Float64=1e20)

Wrapper for a subset of Ipopt options to set during parameter estimation.

For more information about each options see the Ipopt [documentation](https://coin-or.github.io/Ipopt/OPTIONS.html)

## Arguments
- `print_level`: Output verbosity level (valid values are 0 ≤ print_level ≤ 12)
- `max_iter`: Maximum number of iterations
- `tol`: Relative convergence tolerance
- `acceptable_tol`: Acceptable relative convergence tolerance
- `max_wall_time`: Max wall time optimisation is allowed to run
- `acceptable_obj_change_tol`: Acceptance stopping criterion based on objective function change.

See also [`calibrate_model`](@ref) and [`calibrate_multistart`](@ref).
"""
struct IpoptOptions
    print_level::Int64
    max_iter::Int64
    tol::Float64
    acceptable_tol::Float64
    max_wall_time::Float64
    acceptable_obj_change_tol::Float64
end
function IpoptOptions(; print_level::Int64 = 0,
                      max_iter::Int64 = 1000,
                      tol::Float64 = 1e-8,
                      acceptable_tol::Float64 = 1e-6,
                      max_wall_time::Float64 = 1e20,
                      acceptable_obj_change_tol::Float64 = 1e20)
    return IpoptOptions(print_level, max_iter, tol, acceptable_tol, max_wall_time,
                        acceptable_obj_change_tol)
end

struct PEtabOptimisationResult
    alg::Symbol
    xtrace::Vector{Vector{Float64}}
    ftrace::Vector{Float64}
    niterations::Int64
    fmin::Float64
    x0::ComponentArray{Float64}
    xmin::ComponentArray{Float64}
    converged
    runtime::Float64
    original
end

"""
    PEtabMultistartResult(dirres::String; which_run::String="1")

Read PEtab multistart optimization results saved at `dirres`.

Each time a new optimization run is performed, results are saved with unique numerical endings
appended to the directory specified by `dirres`. Results from a specific run can be retreived
by specifying the numerical ending by `which_run`. For example, to access results from the second run,
set `which_run="2"`.
"""
struct PEtabMultistartResult
    xmin::ComponentArray{Float64} # Parameter vectors (if user wants to save them)
    fmin::Float64 # Likelihood value (if user wants to save them)
    nmultistarts::Int
    alg::Symbol
    multistart_method::String
    dirsave::Union{String, Nothing}
    runs::Vector{PEtabOptimisationResult} # See above
end
function PEtabMultistartResult(dirres::String; which_run::Integer = 1)::PEtabMultistartResult
    @assert isdir(dirres) "Directory $dirres does not exist"

    i = which_run |> string
    path_res = joinpath(dirres, "results$i.csv")
    path_parameters = joinpath(dirres, "xmins$i.csv")
    path_startguess = joinpath(dirres, "startguesses$i.csv")
    path_trace = joinpath(dirres, "trace$i.csv")
    @assert isfile(path_res) "Result file $(path_res) does not exist"
    @assert isfile(path_parameters) "Optimal parameters file $(path_parameters) does not exist"
    @assert isfile(path_startguess) "Startguess file $(path_startguess) does not exist"

    res_df = CSV.read(path_res, DataFrame)
    x_df = CSV.read(path_parameters, DataFrame)
    startguesses_df = CSV.read(path_startguess, DataFrame)
    if isfile(path_trace)
        trace_df = CSV.read(path_trace, DataFrame)
    end
    runs = Vector{PEtab.PEtabOptimisationResult}(undef, nrow(res_df))
    for i in eachindex(runs)
        if isfile(path_trace)
            trace_df_i = trace_df[findall(x -> x == i, trace_df[!, :startguess]), :]
            _ftrace = trace_df_i[!, :ftrace]
            _xtrace = [Vector{Float64}(trace_df_i[i, 1:(end - 2)])
                       for i in 1:size(trace_df_i)[1]]
        else
            _ftrace = Vector{Float64}(undef, 0)
            _xtrace = Vector{Vector{Float64}}(undef, 0)
        end
        xnames = propertynames(x_df)[1:end-1] |> collect
        xmin = ComponentArray(; (xnames .=> x_df[i, 1:(end - 1)] |> Vector{Float64})...)
        xstart = ComponentArray(; (xnames .=> startguesses_df[i, 1:(end - 1)] |> Vector{Float64})...)
        runs[i] = PEtabOptimisationResult(Symbol(res_df[i, :alg]), _xtrace, _ftrace,
                                          res_df[i, :niterations], res_df[i, :fmin],
                                          xstart, xmin, res_df[i, :converged],
                                          res_df[i, :runtime], nothing)
    end
    bestrun = runs[argmin([isnan(r.fmin) ? Inf : r.fmin for r in runs])]
    fmin = bestrun.fmin
    xmin = bestrun.xmin
    nmultistarts = length(runs)
    return PEtabMultistartResult(xmin, fmin, nmultistarts, bestrun.alg, "", dirres, runs)
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

struct InferenceInfo{d1 <: Vector{<:Distribution{Univariate, Continuous}},
                     d2 <: Vector{<:Distribution{Univariate, Continuous}},
                     b1,
                     b2}
    priors::d1
    tpriors::d2
    bijectors::b1
    inv_bijectors::b2
    priors_scale::Vector{Symbol}
    parameters_scale::Vector{Symbol}
    parameters_id::Vector{Symbol}
end

"""
    PEtabLogDensity(prob::PEtabODEProblem)

Construct a `LogDensityProblem` using the likelihood and gradient method from the `PEtabODEProblem`.

This LogDensityProblem method defines everything needed to perform Bayesian inference
with libraries such as `AdvancedHMC.jl` (which includes algorithms like NUTS, used by `Turing.jl`),
`AdaptiveMCMC.jl` for adaptive Markov Chain Monte Carlo methods, and `Pigeon.jl` for parallel tempering
methods. For examples on how to perform inference, see the documentation.
"""
struct PEtabLogDensity{T <: InferenceInfo,
                       I <: Integer,
                       T2 <: AbstractFloat}
    inference_info::T
    logtarget::Any
    logtarget_gradient::Any
    initial_value::Vector{T2}
    dim::I
end
function (logpotential::PEtab.PEtabLogDensity)(x)
    return logpotential.logtarget(x)
end
