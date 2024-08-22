function PEtab.create_gradient_function(which_method::Symbol,
                                        ode_problem::ODEProblem,
                                        ode_solver::ODESolver,
                                        ss_solver::SteadyStateSolver,
                                        petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                        petab_ODESolver_cache::PEtab.PEtabODESolverCache,
                                        petab_model::PEtabModel,
                                        simulation_info::PEtab.SimulationInfo,
                                        θ_indices::PEtab.ParameterIndices,
                                        measurement_info::PEtab.MeasurementsInfo,
                                        parameter_info::PEtab.ParametersInfo,
                                        sensealg::Union{InterpolatingAdjoint,
                                                        QuadratureAdjoint,
                                                        GaussAdjoint},
                                        prior_info::PEtab.PriorInfo;
                                        chunksize::Union{Nothing, Int64} = nothing,
                                        sensealg_ss = nothing,
                                        split_over_conditions::Bool = false)
    # Fast but numerically unstable method
    if simulation_info.has_pre_equilibration == true &&
       typeof(sensealg_ss) <: SteadyStateAdjoint
        @warn "If using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient sensealg_ss is as provided SteadyStateAdjoint. However, SteadyStateAdjoint fails if the Jacobian is singular hence we recomend you check that the Jacobian is non-singular."
    end
    _sensealg_ss = isnothing(sensealg_ss) ? sensealg : sensealg_ss
    # If sensealg_ss = GaussAdjoint as we do not actually have any observations during the
    # pre-eq simulations, there is no difference between using Guass and Interpolating
    # adjoint. Hence, to keep the size of the code-base smaller we use Gauss-adjoint
    if _sensealg_ss isa GaussAdjoint
        _sensealg_ss = InterpolatingAdjoint(autojacvec = _sensealg_ss.autojacvec)
    end

    iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = PEtab.get_index_parameters_not_ODE(θ_indices)
    compute_cost_θ_not_ODE = let iθ_sd = iθ_sd, iθ_observable = iθ_observable,
        iθ_non_dynamic = iθ_non_dynamic,
        petab_model = petab_model, simulation_info = simulation_info, θ_indices = θ_indices,
        measurement_info = measurement_info, parameter_info = parameter_info,
        petab_ODE_cache = petab_ODE_cache

        (x) -> PEtab.compute_cost_not_solve_ODE(x[iθ_sd],
                                                x[iθ_observable],
                                                x[iθ_non_dynamic],
                                                petab_model,
                                                simulation_info,
                                                θ_indices,
                                                measurement_info,
                                                parameter_info,
                                                petab_ODE_cache,
                                                exp_id_solve = [:all],
                                                compute_gradient_not_solve_adjoint = true)
    end

    _compute_gradient! = let ode_solver = ode_solver, ss_solver = ss_solver,
        compute_cost_θ_not_ODE = compute_cost_θ_not_ODE,
        sensealg = sensealg, _sensealg_ss = _sensealg_ss, ode_problem = ode_problem,
        petab_model = petab_model,
        simulation_info = simulation_info, θ_indices = θ_indices,
        measurement_info = measurement_info,
        parameter_info = parameter_info, prior_info = prior_info,
        petab_ODE_cache = petab_ODE_cache,
        petab_ODESolver_cache = petab_ODESolver_cache

        (gradient, θ) -> compute_gradient_adjoint!(gradient,
                                                   θ,
                                                   ode_solver,
                                                   ss_solver,
                                                   compute_cost_θ_not_ODE,
                                                   sensealg,
                                                   _sensealg_ss,
                                                   ode_problem,
                                                   petab_model,
                                                   simulation_info,
                                                   θ_indices,
                                                   measurement_info,
                                                   parameter_info,
                                                   prior_info,
                                                   petab_ODE_cache,
                                                   petab_ODESolver_cache,
                                                   exp_id_solve = [:all])
    end

    compute_gradient = let _compute_gradient! = _compute_gradient!
        (θ) -> begin
            gradient = zeros(Float64, length(θ))
            _compute_gradient!(gradient, θ)
            return gradient
        end
    end

    return _compute_gradient!, compute_gradient
end

function PEtab.set_sensealg(sensealg, ::Val{:Adjoint})
    if !isnothing(sensealg)
        @assert any(typeof(sensealg) .<:
                    [InterpolatingAdjoint, QuadratureAdjoint, GaussAdjoint]) "For gradient method :Adjoint allowed sensealg args are InterpolatingAdjoint, GaussAdjoint, QuadratureAdjoint not $sensealg"
        return sensealg
    end

    return InterpolatingAdjoint(autojacvec = ReverseDiffVJP())
end
function PEtab.set_sensealg(sensealg::Union{ForwardSensitivity, ForwardDiffSensitivity},
                            ::Val{:ForwardEquations})
    return sensealg
end

function PEtab.get_callbackset(ode_problem::ODEProblem,
                               simulation_info::PEtab.SimulationInfo,
                               simulation_condition_id::Symbol,
                               sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint,
                                               GaussAdjoint})::SciMLBase.DECallback
    cbset = SciMLSensitivity.track_callbacks(simulation_info.callbacks[simulation_condition_id],
                                             ode_problem.tspan[1],
                                             ode_problem.u0, ode_problem.p, sensealg)
    simulation_info.tracked_callbacks[simulation_condition_id] = cbset
    return cbset
end

# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθ_zygote(θ::AbstractVector,
                           n_parameters_estimate::Vector{Symbol},
                           parameter_info::PEtab.ParametersInfo;
                           reverse_transform::Bool = false)::AbstractVector
    iθ = [findfirst(x -> x == n_parameters_estimate[i], parameter_info.parameter_id)
          for i in eachindex(n_parameters_estimate)]
    should_transform = [parameter_info.parameter_scale[i] == :log10 ? true : false
                        for i in iθ]
    should_not_transform = .!should_transform

    if reverse_transform == false
        out = exp10.(θ) .* should_transform .+ θ .* should_not_transform
    else
        out = log10.(θ) .* should_transform .+ θ .* should_not_transform
    end
    return out
end

function PEtab.create_gradient_function(::Val{:Zygote},
                                        ode_problem::ODEProblem,
                                        ode_solver::ODESolver,
                                        ss_solver::SteadyStateSolver,
                                        petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                        petab_ODESolver_cache::PEtab.PEtabODESolverCache,
                                        petab_model::PEtabModel,
                                        simulation_info::PEtab.SimulationInfo,
                                        θ_indices::PEtab.ParameterIndices,
                                        measurement_info::PEtab.MeasurementsInfo,
                                        parameter_info::PEtab.ParametersInfo,
                                        sensealg::SciMLBase.AbstractSensitivityAlgorithm,
                                        prior_info::PEtab.PriorInfo;
                                        chunksize::Union{Nothing, Int64} = nothing,
                                        sensealg_ss = nothing,
                                        split_over_conditions::Bool = false)
    change_simulation_condition = (p_ode_problem, u0, conditionId, θ_dynamic) -> PEtab._change_simulation_condition(p_ode_problem,
                                                                                                                    u0,
                                                                                                                    conditionId,
                                                                                                                    θ_dynamic,
                                                                                                                    petab_model,
                                                                                                                    θ_indices)
    solve_ode_condition = (ode_problem, conditionId, θ_dynamic, tmax) -> solve_ode_condition_zygote(ode_problem,
                                                                                                    conditionId,
                                                                                                    θ_dynamic,
                                                                                                    tmax,
                                                                                                    change_simulation_condition,
                                                                                                    measurement_info,
                                                                                                    simulation_info,
                                                                                                    ode_solver.solver,
                                                                                                    ode_solver.abstol,
                                                                                                    ode_solver.reltol,
                                                                                                    ss_solver.abstol,
                                                                                                    ss_solver.reltol,
                                                                                                    sensealg,
                                                                                                    petab_model.compute_tstops)
    _compute_gradient! = (gradient, θ_est) -> compute_gradient_zygote!(gradient,
                                                                       θ_est,
                                                                       ode_problem,
                                                                       petab_model,
                                                                       simulation_info,
                                                                       θ_indices,
                                                                       measurement_info,
                                                                       parameter_info,
                                                                       solve_ode_condition,
                                                                       prior_info,
                                                                       petab_ODE_cache)

    compute_gradient = let _compute_gradient! = _compute_gradient!
        (θ) -> begin
            gradient = zeros(Float64, length(θ))
            _compute_gradient!(gradient, θ)
            return gradient
        end
    end

    return _compute_gradient!, compute_gradient
end

function PEtab.create_cost_function(::Val{:Zygote},
                                    ode_problem::ODEProblem,
                                    ode_solver::ODESolver,
                                    ss_solver::SteadyStateSolver,
                                    petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                    petab_ODESolver_cache::PEtab.PEtabODESolverCache,
                                    petab_model::PEtabModel,
                                    simulation_info::PEtab.SimulationInfo,
                                    θ_indices::PEtab.ParameterIndices,
                                    measurement_info::PEtab.MeasurementsInfo,
                                    parameter_info::PEtab.ParametersInfo,
                                    prior_info::PEtab.PriorInfo,
                                    sensealg,
                                    compute_residuals)
    change_simulation_condition = (p_ode_problem, u0, conditionId, θ_dynamic) -> PEtab._change_simulation_condition(p_ode_problem,
                                                                                                                    u0,
                                                                                                                    conditionId,
                                                                                                                    θ_dynamic,
                                                                                                                    petab_model,
                                                                                                                    θ_indices)
    solve_ode_condition = (ode_problem, conditionId, θ_dynamic, tmax) -> solve_ode_condition_zygote(ode_problem,
                                                                                                    conditionId,
                                                                                                    θ_dynamic,
                                                                                                    tmax,
                                                                                                    change_simulation_condition,
                                                                                                    measurement_info,
                                                                                                    simulation_info,
                                                                                                    ode_solver.solver,
                                                                                                    ode_solver.abstol,
                                                                                                    ode_solver.reltol,
                                                                                                    ss_solver.abstol,
                                                                                                    ss_solver.reltol,
                                                                                                    sensealg,
                                                                                                    petab_model.compute_tstops)
    __compute_cost = (θ_est) -> compute_cost_zygote(θ_est,
                                                    ode_problem,
                                                    petab_model,
                                                    simulation_info,
                                                    θ_indices,
                                                    measurement_info,
                                                    parameter_info,
                                                    solve_ode_condition,
                                                    prior_info)

    return __compute_cost
end

function PEtab.set_sensealg(sensealg, ::Val{:Zygote})
    if !isnothing(sensealg)
        @assert (typeof(sensealg)<:SciMLSensitivity.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
        return sensealg
    end

    return SciMLSensitivity.ForwardSensitivity()
end
