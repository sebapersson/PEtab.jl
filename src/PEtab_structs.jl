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
    abstol::Float64
    reltol::Float64
    force_dtmin::Bool
    dtmin::Union{Float64, Nothing}
    maxiters::Int64
    verbose::Bool
end
function ODESolver(solver::T1;
                   abstol::Float64=1e-8,
                   reltol::Float64=1e-8,
                   force_dtmin::Bool=false,
                   dtmin::Union{Float64, Nothing}=nothing,
                   maxiters::Int64=Int64(1e4), 
                   verbose::Bool=true) where T1 <: SciMLAlgorithm

    return ODESolver(solver, abstol, reltol, force_dtmin, dtmin, maxiters, verbose)
end


"""
    SteadyStateSolver(method::Symbol;
                      check_simulation_steady_state::Symbol=:wrms,
                      rootfinding_alg=nothing,
                      abstol=nothing,
                      reltol=nothing,
                      maxiters=nothing)

Setup options for finding steady-state via either `method=:Rootfinding` or `method=:Simulate`.

For `method=:Rootfinding`, the steady-state `u*` is found by solving the problem `du = f(u, p, t) ≈ 0` with tolerances 
`abstol` and `reltol` via an automatically chosen optimization algorithm (`rootfinding_alg=nothing`) or via any 
provided algorithm in [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl).

For `method=:Simulate`, the steady-state `u*` is found by simulating the ODE system until `du = f(u, p, t) ≈ 0`. 
Two options are available for `check_simulation_steady_state`:
- `:wrms` : Weighted root-mean square √(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1
- `:Newton` : If Newton-step `Δu` is sufficiently small √(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1.
        - Newton often performs better but requires an invertible Jacobian. In case it's not fulfilled, the code 
          switches automatically to `:wrms`.

`maxiters` refers to either the maximum number of rootfinding steps or the maximum number of integration steps, 
depending on the chosen method.
"""
struct SteadyStateSolver{T1 <: Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm},
                         T2 <: Union{Nothing, AbstractFloat},
                         T3 <: Union{Nothing, NonlinearProblem},
                         CA <: Union{Nothing, SciMLBase.DECallback},
                         T4 <: Union{Nothing, Integer}}
    method::Symbol
    rootfinding_alg::T1
    check_simulation_steady_state::Symbol
    abstol::T2
    reltol::T2
    maxiters::T4
    callback_ss::CA
    nonlinearsolve_problem::T3
end
function SteadyStateSolver(method::Symbol;
                           check_simulation_steady_state::Symbol=:wrms,
                           rootfinding_alg::Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm}=nothing,
                           abstol=nothing,
                           reltol=nothing,
                           maxiters::Union{Nothing, Int64}=nothing)::SteadyStateSolver

    @assert method ∈ [:Rootfinding, :Simulate] "Method used to find steady state can either be :Rootfinding or :Simulate not $method"

    if method === :Simulate
        return _get_steady_state_solver(check_simulation_steady_state, abstol, reltol, maxiters)
    else
        return _get_steady_state_solver(rootfinding_alg, abstol, reltol, maxiters)
    end
end


struct SimulationInfo{T1<:Dict{<:Symbol, <:SciMLBase.DECallback},
                      T2<:Dict{<:Symbol, <:SciMLBase.DECallback}}

    pre_equilibration_condition_id::Vector{Symbol}
    simulation_condition_id::Vector{Symbol}
    experimental_condition_id::Vector{Symbol}
    has_pre_equilibration_condition_id::Bool
    ode_sols::Dict{Symbol, Union{Nothing, ODESolution}}
    ode_sols_derivatives::Dict{Symbol, Union{Nothing, ODESolution}}
    ode_sols_pre_equlibrium::Dict{Symbol, Union{Nothing, ODESolution, SciMLBase.NonlinearSolution}}
    could_solve::Vector{Bool}
    tmax::Dict{Symbol, Float64}
    time_observed::Dict{Symbol, Vector{Float64}}
    i_measurements::Dict{Symbol, Vector{Int64}}
    i_time_ode_sol::Vector{Int64}
    i_per_time_point::Dict{Symbol, Vector{Vector{Int64}}}
    time_position_ode_sol::Dict{Symbol, UnitRange{Int64}}
    callbacks::T1
    tracked_callbacks::T2
    sensealg
end


struct θObsOrSdParameterMap
    should_estimate::Array{Bool, 1}
    index_in_θ::Array{Int64, 1}
    constant_values::Vector{Float64}
    n_parameters::Int64
    is_single_constant::Bool
end


struct MapConditionId
    constant_parameters::Vector{Float64}
    i_ode_constant_parameters::Vector{Int64}
    constant_states::Vector{Float64}
    i_ode_constant_states::Vector{Int64}
    iθ_dynamic::Vector{Int64}
    i_ode_problem_θ_dynamic::Vector{Int64}
end


struct Map_ode_problem
    iθ_dynamic::Vector{Int64}
    i_ode_problem_θ_dynamic::Vector{Int64}
end


struct ParameterIndices

    iθ_dynamic::Vector{Int64}
    iθ_observable::Vector{Int64}
    iθ_sd::Vector{Int64}
    iθ_non_dynamic::Vector{Int64}
    iθ_not_ode::Vector{Int64}
    θ_dynamic_names::Vector{Symbol}
    θ_observable_names::Vector{Symbol}
    θ_sd_names::Vector{Symbol}
    θ_non_dynamic_names::Vector{Symbol}
    θ_not_odeNames::Vector{Symbol}
    θ_names::Vector{Symbol}
    θ_scale::Dict{Symbol, Symbol}
    mapθ_observable::Vector{θObsOrSdParameterMap}
    mapθ_sd::Vector{θObsOrSdParameterMap}
    map_ode_problem::Map_ode_problem
    maps_conidition_id::Dict{<:Symbol, <:MapConditionId}
end


struct PriorInfo
    logpdf::Dict{Symbol, Function}
    distribution::Dict{Symbol, Distribution{Univariate, Continuous}}
    initialisation_distribution::Dict{Symbol, Distribution{Univariate, Continuous}}
    prior_on_parameter_scale::Dict{<:Symbol, <:Bool}
    has_priors::Bool
end


"""
    PEtabModel

A Julia-compatible representation of a PEtab-specified problem.

For constructor see below.

!!! note
    Most of the functions in `PEtabModel` are not intended to be accessed by the user. For example, `compute_h` 
    (and similar functions) require indices that are built in the background to efficiently map parameters between 
    experimental (simulation) conditions. Rather, `PEtabModel` holds all information needed to create a 
    `PEtabODEProblem`, and in the future, `PEtabSDEProblem`, etc.

# Fields
- `model_name`: The model name extracted from the PEtab YAML file.
- `compute_h`: Computes the observable `h` for a specific time point and simulation condition.
- `compute_u0!`: Computes in-place initial values using `ODEProblem.p` for a simulation condition; `compute_u0!(u0, p)`.
- `compute_u0`: Computes initial values as above, but not in-place; `u0 = compute_u0(p)`.
- `compute_σ`: Computes the noise parameter `σ` for a specific time point and simulation condition.
- `compute_∂h∂u!`: Computes the gradient of `h` with respect to `ODEModel` states (`u`) for a specific time point and simulation condition.
- `compute_∂σ∂u!`: Computes the gradient of `σ` with respect to `ODEModel` states (`u`) for a specific time point and simulation condition.
- `compute_∂h∂p!`: Computes the gradient of `h` with respect to `ODEProblem.p`.
- `compute_∂σ∂p!`: Computes the gradient of `σ` with respect to `ODEProblem.p`.
- `compute_tstops`: Computes the event times in case the model has `DiscreteCallbacks` (events).
- `convert_tspan::Bool`: Tracks whether the time span should be converted to `Dual` numbers for `ForwardDiff.jl` gradients, in case the model has `DiscreteCallbacks` and the trigger time is a parameter set to be estimated.
- `dir_model`: The directory where the model.xml and PEtab files are stored.
- `dir_julia`: The directory where the Julia-model files created by parsing the PEtab files (e.g., SBML file) are stored.
- `ode_system`: A `ModellingToolkit.jl` ODE system obtained from parsing the model SBML file.
- `parameter_map`: A `ModellingToolkit.jl` parameter map for the ODE system.
- `state_map`: A `ModellingToolkit.jl` state map for the ODE system describing how the initial values are computed, e.g., whether or not certain initial values are computed from parameters in the `parameter_map`.
- `parameter_names`: The names of the parameters in the `ode_system`.
- `state_names`: The names of the states in the `ode_system`.
- `path_measurements`: The path to the PEtab measurements file.
- `path_conditions`: The path to the PEtab conditions file.
- `path_observables`: The path to the PEtab observables file.
- `path_parameters`: The path to the PEtab parameters file.
- `path_SBML`: The path to the PEtab SBML file.
- `path_yaml`: The path to the PEtab YAML file.
- `model_callbacks`: This stores potential model callbacks or events.
- `check_callback_is_active`: Piecewise SBML statements are transformed to DiscreteCallbacks that are activated at a specific time-point. The piecewise callback has a default value at t0 and is only triggered when reaching t_activation. If t_activation ≤ 0 (never reached when solving the model), this function checks whether the callback should be triggered before solving the model.

## Constructor

    PEtabModel(path_yaml::String;
               build_julia_files::Bool=false,
               verbose::Bool=true,
               ifelse_to_event::Bool=true,
               write_to_file::Bool=true,
               jlfile_path::String="")::PEtabModel

Create a PEtabModel from a PEtab specified problem with a YAML-file located at `path_yaml`.

When parsing a PEtab problem, several things happen under the hood:

1. The SBML file is translated into `ModelingToolkit.jl` format to allow for symbolic computations of the ODE-model Jacobian. Piecewise and model events are further written into `DifferentialEquations.jl` callbacks.
2. The observable PEtab table is translated into a Julia file with functions for computing the observable (`h`), noise parameter (`σ`), and initial values (`u0`).
3. To allow gradients via adjoint sensitivity analysis and/or forward sensitivity equations, the gradients of `h` and `σ` are computed symbolically with respect to the ODE model's states (`u`) and parameters (`ode_problem.p`).

All of this happens automatically, and resulting files are stored under `petab_model.dir_julia` assuming write_to_file=true. To save time, `forceBuildJlFiles=false` by default, which means that Julia files are not rebuilt if they already exist.

# Arguments
- `path_yaml::String`: Path to the PEtab problem YAML file.
- `build_julia_files::Bool=false`: If `true`, forces the creation of Julia files for the problem even if they already exist.
- `verbose::Bool=true`: If `true`, displays verbose output during parsing.
- `ifelse_to_event::Bool=true`: If `true`, rewrites `if-else` statements in the SBML model as event-based callbacks.
- `write_to_file::Bool=true`: If `true`, writes built Julia files to disk (recomended)

# Example
```julia
petab_model = PEtabModel("path_to_petab_problem_yaml")
```
"""
struct PEtabModel{F1<:Function,
                  F2<:Function,
                  F3<:Function,
                  F4<:Function,
                  F5<:Function,
                  F6<:Function,
                  F7<:Function,
                  F8<:Function,
                  F9<:Function,
                  C<:SciMLBase.DECallback,
                  FA<:Vector{<:Function}, 
                  S}
    model_name::String
    compute_h::F1
    compute_u0!::F2
    compute_u0::F3
    compute_σ::F4
    compute_∂h∂u!::F5
    compute_∂σ∂u!::F6
    compute_∂h∂p!::F7
    compute_∂σ∂p!::F8
    compute_tstops::F9
    convert_tspan::Bool
    system::S
    parameter_map
    state_map
    parameter_names
    state_names
    dir_model::String
    dir_julia::String
    path_measurements::CSV.File
    path_conditions::CSV.File
    path_observables::CSV.File
    path_parameters::CSV.File
    path_SBML::String
    path_yaml::String
    model_callbacks::C
    check_callback_is_active::FA
    defined_in_julia::Bool
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
- `compute_gradient!`: For θ computes in-place gradient compute_gradient!(gradient, θ)
- `compute_gradient`: For θ computes out-place gradient gradient = compute_gradient(θ)
- `compute_hessian!`: For θ computes in-place hessian-(approximation) compute_hessian!(hessian, θ)
- `compute_hessian`: For θ computes out-place hessian-(approximation) hessian = compute_hessian(θ)
- `compute_FIM!`: For θ computes the empirical Fisher-Information-Matrix (FIM) which is the Hessian of the negative-log-likelihood  compute_FIM!(FIM, θ).
- `compute_FIM`: For θ computes FIM out of place FIM = compute_FIM(θ).
- `compute_simulated_values`: For θ compute the corresponding model (simulated) values to the measurements in the same order as in the Measurements PEtab table
- `compute_residuals`: For θ compute the residuals (h_model - h_observed)^2 / σ^2 in the same order as in the Measurements PEtab table
- `gradient_method`: The method used to compute the gradient (either :ForwardDiff, :ForwardEquations, :Adjoint, or :Zygote).
- `hessian_method`: The method used to compute or approximate the Hessian (either :ForwardDiff, :BlocForwardDiff, or :GaussNewton).
- `FIM_method`: The method used to compute FIM, either :ForwardDiff (full Hessian) or :GaussNewton (only recomended for >100 parameter models)
- `n_parameters_esimtate`: The number of parameters to estimate.
- `θ_names`: The names of the parameters in θ.
- `θ_nominal`: The nominal values of θ as specified in the PEtab parameters file.
- `θ_nominalT`: The nominal values of θ on the parameter scale (e.g., log) as specified in the PEtab parameters file.
- `lower_bounds`: The lower parameter bounds on the parameter scale for θ as specified in the PEtab parameters file.
- `upper_bounds`: The upper parameter bounds on the parameter scale for θ as specified in the PEtab parameters file.
- `petab_model`: The PEtabModel used to construct the PEtabODEProblem.
- `ode_solver`: The options for the ODE solver specified when creating the PEtabODEProblem.
- `ode_solver_gradient`: The options for the ODE solver gradient specified when creating the PEtabODEProblem.

## Constructor

    PEtabODEProblem(petab_model::PEtabModel; <keyword arguments>)

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
struct PEtabODEProblem{F1<:Function,
                       F2<:Function,
                       F3<:Function,
                       F4<:Function,
                       F5<:Function, 
                       F6<:Function, 
                       F7<:Function}

    compute_cost::F1
    compute_chi2
    compute_gradient!::F2
    compute_gradient::F3
    compute_hessian!::F4
    compute_hessian::F5
    compute_FIM!::F6
    compute_FIM::F7
    compute_simulated_values
    compute_residuals
    cost_method::Symbol
    gradient_method::Symbol
    hessian_method::Union{Symbol, Nothing}
    FIM_method::Symbol
    n_parameters_esimtate::Int64
    θ_names::Vector{Symbol}
    θ_nominal::Vector{Float64}
    θ_nominalT::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    petab_model::PEtabModel
    ode_solver::ODESolver
    ode_solver_gradient::ODESolver
    ss_solver::SteadyStateSolver
    ss_solver_gradient::SteadyStateSolver
    θ_indices::ParameterIndices
    simulation_info::SimulationInfo
    ode_problem::ODEProblem
    split_over_conditions::Bool
    prior_info::PriorInfo
end


struct ParametersInfo
    nominal_value::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    parameter_id::Vector{Symbol}
    parameter_scale::Vector{Symbol}
    estimate::Vector{Bool}
    n_parameters_esimtate::Int64
end


struct MeasurementsInfo{T<:Vector{<:Union{<:String, <:AbstractFloat}}}

    measurement::Vector{Float64}
    measurementT::Vector{Float64}
    simulated_values::Vector{Float64}
    chi2_values::Vector{Float64}
    residuals::Vector{Float64}
    measurement_transformation::Vector{Symbol}
    time::Vector{Float64}
    observable_id::Vector{Symbol}
    pre_equilibration_condition_id::Vector{Symbol}
    simulation_condition_id::Vector{Symbol}
    noise_parameters::T
    observable_parameters::Vector{String}
end


struct PEtabODEProblemCache{T1 <: AbstractVector,
                            T2 <: DiffCache,
                            T3 <: AbstractVector,
                            T4 <: AbstractMatrix}
    θ_dynamic::T1
    θ_sd::T1
    θ_observable::T1
    θ_non_dynamic::T1
    θ_dynamicT::T2 # T = transformed vector
    θ_sdT::T2
    θ_observableT::T2
    θ_non_dynamicT::T2
    gradient_θ_dyanmic::T1
    gradient_θ_not_ode::T1
    jacobian_gn::T4
    residuals_gn::T1
    _gradient::T1
    _gradient_adjoint::T1
    S_t0::T4
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
    sol_values::T4
    θ_dynamic_input_order::Vector{Int64}
    θ_dynamic_output_order::Vector{Int64}
    nθ_dynamic::Vector{Int64}
end


struct PEtabODESolverCache
    p_ode_problem_cache
    u0_cache
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
function Fides(hessian_method::Union{Nothing, Symbol}; verbose::Bool=false)
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

See also [`calibrate_model`](@ref) and [`calibrate_model_multistart`](@ref).

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

See also [`calibrate_model`](@ref) and [`calibrate_model_multistart`](@ref).
"""
struct IpoptOptions
    print_level::Int64
    max_iter::Int64
    tol::Float64
    acceptable_tol::Float64
    max_wall_time::Float64
    acceptable_obj_change_tol::Float64
end
function IpoptOptions(;print_level::Int64=0, 
                      max_iter::Int64=1000, 
                      tol::Float64=1e-8, 
                      acceptable_tol::Float64=1e-6, 
                      max_wall_time::Float64=1e20, 
                      acceptable_obj_change_tol::Float64=1e20)

    return IpoptOptions(print_level, max_iter, tol, acceptable_tol, max_wall_time, acceptable_obj_change_tol)
end


struct PEtabOptimisationResult{T<:Any}
    alg::Symbol
    xtrace::Vector{Vector{Float64}} # Parameter vectors (if user wants to save them)
    ftrace::Vector{Float64} # Likelihood value (if user wants to save them)
    n_iterations::Int64 # Number of iterations optimiser
    fmin::Float64 # Best optimised value 
    x0::Vector{Float64} # Starting point 
    xmin::Vector{Float64} # Last parameter value 
    xnames::Vector{Symbol}
    converged::T # If user wants to 
    runtime::Float64 # Always fun :)
end


"""
    PEtabMultistartOptimisationResult(dir_res::String; which_run::String="1")

Read PEtab multistart optimization results saved at `dir_res`.

Each time a new optimization run is performed, results are saved with unique numerical endings 
appended to the directory specified by `dir_res`. Results from a specific run can be retreived 
by specifying the numerical ending by `which_run`. For example, to access results from the second run, 
set `which_run="2"`.
"""
struct PEtabMultistartOptimisationResult
    xmin::Vector{Float64} # Parameter vectors (if user wants to save them)
    xnames::Vector{Symbol}
    fmin::Float64 # Likelihood value (if user wants to save them)
    n_multistarts::Int
    alg::Symbol
    multistart_method::String
    dir_save::Union{String, Nothing}
    runs::Vector{PEtabOptimisationResult} # See above
end
function PEtabMultistartOptimisationResult(dir_res::String; which_run::String="1")::PEtabMultistartOptimisationResult
    
    @assert isdir(dir_res) "Directory $dir_res does not exist"

    path_res = joinpath(dir_res, "Optimisation_results" * which_run * ".csv")
    path_parameters = joinpath(dir_res, "Best_parameters" * which_run * ".csv")
    path_startguess = joinpath(dir_res, "Start_guesses" * which_run * ".csv")
    path_trace = joinpath(dir_res, "Trace" * which_run * ".csv")
    
    @assert isfile(path_res) "Result file (Optimisation_results...) does not exist"
    @assert isfile(path_parameters) "Optimal parameters file (Best_parameters...) does not exist"
    @assert isfile(path_startguess) "Startguess file (Start_guesses...) does not exist"
    data_res = CSV.read(path_res, DataFrame)
    data_parameters = CSV.read(path_parameters, DataFrame)
    data_startguess = CSV.read(path_startguess, DataFrame)
    xnames = Symbol.(names(data_parameters))[1:end-1]

    if isfile(path_trace)
        data_trace = CSV.read(path_trace, DataFrame)
    end

    _runs = Vector{PEtab.PEtabOptimisationResult}(undef, size(data_res)[1])
    for i in eachindex(_runs)

        if isfile(path_trace)
            data_trace_i = data_trace[findall(x -> x == i, data_trace[!, :Start_guess]), :]
            _ftrace = data_trace_i[!, :f_trace]
            _xtrace = [Vector{Float64}(data_trace_i[i, 1:end-2]) for i in 1:size(data_trace_i)[1]]
        else
            _ftrace = Vector{Float64}(undef, 0)
            _xtrace = Vector{Vector{Float64}}(undef, 0)
        end

        _runs[i] = PEtab.PEtabOptimisationResult(Symbol(data_res[i, :alg]), 
                                                 _xtrace,
                                                 _ftrace,
                                                 data_res[i, :n_iterations],
                                                 data_res[i, :fmin],
                                                 data_startguess[i, 1:end-1] |> Vector{Float64},
                                                 data_parameters[i, 1:end-1] |> Vector{Float64},
                                                 xnames,
                                                 data_res[i, :converged], 
                                                 data_res[i, :run_time])
    end
    run_best = _runs[argmin([isnan(_runs[i].fmin) ? Inf : _runs[i].fmin for i in eachindex(_runs)])]
    _res = PEtabMultistartOptimisationResult(run_best.xmin,
                                             xnames,
                                             run_best.fmin,
                                             length(_runs),
                                             run_best.alg,
                                             "",
                                             dir_res,
                                             _runs)
    return _res
end


"""
    PEtabEvent(condition, affect, target)

An event triggered by `condition` that sets the value of `target` to that of `affect`.
    
If `condition` is a single value or model parameter (e.g., `c1` or `1.0`), the event is triggered when 
time reaches that value (e.g., `t == c1` or `t == 1.0`). Condition can also depend on model states, 
for example, `S == 2.0` will trigger the event when the state `S` reaches the value 2.0. In contrast, 
`S > 2.0` will trigger the condition when `S` increases from below 2.0 (specifically, the event is 
triggered when the condition changes from `false` to `true`). Note that the condition can contain 
model parameter values or species, e.g., `S > c1`.
    
`affect` can be a constant value (e.g., `1.0`) or an algebraic expression of model parameters/states. 
For example, to add `5.0` to the state `S`, write `S + 5`. In case an event affects several parameters
and/or states provide affect as a `Vector`, for example `[S + 5, 1.0]`.
    
`target` is either a model state or parameter that the event acts on. In case an event affects several 
states and/or parameters provide as a `Vector` where `target[i]` is the target of `affect[i]`.
    
For more details, see the documentation.

!!! note
    If the condition and target are single parameters or states, they can be specied as `Num` (from unpack) or `Symbol`. 
    If the event involves multiple parameters or states, you must provide them as either a `Num` (as shown below) or a 
    `String`.

## Examples 
```julia
using Catalyst
# Trigger event at t = 3.0, add 5 to A
rn = @reaction_network begin
    (k1, k2), A <--> B
end
@unpack A = rn
event = PEtabEvent(3.0, A + 5.0, A)
```
```julia
using Catalyst
# Trigger event at t = k1, set k2 to 3
rn = @reaction_network begin
    (k1, k2), A <--> B
end
event = PEtabEvent(:k1, 3.0, :k2)
```
```julia
using Catalyst
# Trigger event when A == 0.2, set B to 2.0
rn = @reaction_network begin
    (k1, k2), A <--> B
end
@unpack A, B = rn
event = PEtabEvent(A == 0.2, 2.0, B)
```
```julia
using Catalyst
# Trigger event when A == 0.2, set B to 2.0 and A += 2
rn = @reaction_network begin
    (k1, k2), A <--> B
end
@unpack A, B = rn
event = PEtabEvent(A == 0.2, [A + 2, 2.0], [A, B])
```
"""
struct PEtabEvent{T1<:Any, 
                  T2<:Any, 
                  T3<:Any}
    condition::T1
    affect::T2
    target::T3
end


"""
    PEtabObservable(obs_formula, noise_formula; transformation::Symbol=:lin)

Links a model to measurements using an observable formula and measurement noise formula.

The `transformation` argument can take one of three values: `:lin` (for normal measurement noise), `:log`, or `:log10` (for log-normal measurement noise). For a full description of options, including how to define measurement-specific observable and noise parameters, see the main documentation.

## Examples
```julia 
# Example 1: Log-normal measurement noise with known error σ=3.0
@unpack X = rn  # 'rn' is the dynamic model
PEtabObservable(X, 3.0, transformation=:log)
```
```julia 
# Example 2: Normal measurement noise with estimation of σ (defined as PEtabParameter)
@unpack X, Y = rn  # 'rn' is the dynamic model
@parameters sigma
PEtabObservable((X + Y) / X, sigma)
```
```julia
# Example 3: Normal measurement noise with measurement-specific noiseParameter
@unpack X, Y = rn  # 'rn' is the dynamic model
@parameters noiseParameter1  # Must be in the format 'noiseParameter'
PEtabObservable(X, noiseParameter1 * X)
```
"""
struct PEtabObservable
    obs::Any
    transformation::Union{Nothing, Symbol}
    noise_formula::Union{Nothing, Any}
end
function PEtabObservable(obs::Any, 
                         noise_formula::Union{Nothing, Num, T};
                         transformation::Symbol=:lin)::PEtabObservable where T<:Real
    return PEtabObservable(obs, transformation, noise_formula)
end


"""
    PEtabParameter(id::Union{Num, Symbol}; <keyword arguments>)

Represents a parameter to be estimated in a PEtab model calibration problem.

## Keyword Arguments
- `estimate::Bool=true`: Specifies whether the parameter should be estimated (default) or set as constant.
- `value::Union{Nothing, Float64}=nothing`: The parameter value to use if `estimate=false`. Defaults to the midpoint between `lb` and `ub`.
- `scale::Symbol=:log10`: The scale on which to estimate the parameter. Allowed options are `:log10` (default), `:log`, and `:lin`.
- `lb::Float64=1e-3`: The lower parameter bound in parameter estimation (default: 1e-3).
- `ub::Float64=1e-3`: The upper parameter bound in parameter estimation (default: 1e3).
- `prior=nothing`: An optional continuous prior distribution from the Distributions package.
- `prior_on_linear_scale::Bool=true`: Specifies whether the prior is on the linear scale (default) or the transformed scale, e.g., log10-scale.
- `sample_from_prior::Bool=true`: Whether to sample the parameter from the prior distribution when generating startguesses for model calibration.

## Examples
```julia 
# Example 1: Parameter with a Log-Normal prior (LN(μ=3.0, σ=1.0)) estimated on the log10 scale
PEtabParameter(:c1, prior=LogNormal(3.0, 1.0))
```
```julia
# Example 2: Parameter estimated on the log scale with a Normal prior (N(0.0, 1.0)) on the log scale
PEtabParameter(:c1, scale=:log, prior=Normal(0.0, 1.0), prior_on_linear_scale=false)
```
"""
struct PEtabParameter
    parameter::Union{Num, Symbol}
    estimate::Bool
    value::Union{Nothing,Float64}
    lb::Union{Nothing,Float64}
    ub::Union{Nothing,Float64}
    prior::Union{Nothing,Distribution{Univariate, Continuous}}
    prior_on_linear_scale::Bool
    scale::Union{Nothing,Symbol} # :log10, :linear and :log supported.
    sample_from_prior::Bool
end
function PEtabParameter(id::Union{Num, Symbol};
                        estimate::Bool=true,
                        value::Union{Nothing, Float64}=nothing,
                        lb::Union{Nothing, Float64}=1e-3,
                        ub::Union{Nothing, Float64}=1e3,
                        prior::Union{Nothing,Distribution{Univariate, Continuous}}=nothing,
                        prior_on_linear_scale::Bool=true,
                        scale::Union{Nothing, Symbol}=:log10, 
                        sample_from_prior::Bool=true)

    return PEtabParameter(id, estimate, value, lb, ub, prior, prior_on_linear_scale, scale, sample_from_prior)
end



struct PEtabFileError <: Exception
    var::String
end
struct PEtabFormatError <: Exception
    var::String
end
