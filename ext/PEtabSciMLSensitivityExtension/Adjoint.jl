# Compute gradient via adjoint sensitivity analysis
function compute_gradient_adjoint!(gradient::Vector{Float64},
                                   θ_est::Vector{Float64},
                                   solverOptions::ODESolver,
                                   ss_solver::SteadyStateSolver,
                                   compute_cost_θ_not_ODE::Function,
                                   sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                   sensealg_ss::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                   ode_problem::ODEProblem,
                                   petab_model::PEtab.PEtabModel,
                                   simulation_info::PEtab.SimulationInfo,
                                   θ_indices::PEtab.ParameterIndices,
                                   measurement_info::PEtab.MeasurementsInfo,
                                   parameter_info::PEtab.ParametersInfo,
                                   prior_info::PEtab.PriorInfo,
                                   petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                   petab_ODESolver_cache::PEtab.PEtabODESolverCache;
                                   exp_id_solve::Vector{Symbol} = [:all])

    PEtab.splitθ!(θ_est, θ_indices, petab_ODE_cache)
    θ_dynamic = petab_ODE_cache.θ_dynamic
    θ_observable = petab_ODE_cache.θ_observable
    θ_sd = petab_ODE_cache.θ_sd
    θ_non_dynamic = petab_ODE_cache.θ_non_dynamic

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    compute_gradient_adjoint_θ_dynamic!(petab_ODE_cache.gradient_θ_dyanmic, θ_dynamic, θ_sd, θ_observable,  
                                        θ_non_dynamic, ode_problem, solverOptions, ss_solver, sensealg, 
                                        petab_model, simulation_info, θ_indices, measurement_info, parameter_info,
                                        petab_ODE_cache, petab_ODESolver_cache; exp_id_solve=exp_id_solve,
                                        sensealg_ss=sensealg_ss)
    @views gradient[θ_indices.iθ_dynamic] .= petab_ODE_cache.gradient_θ_dyanmic

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if !isempty(petab_ODE_cache.gradient_θ_dyanmic) && all(petab_ODE_cache.gradient_θ_dyanmic .== 0.0)
        gradient .= 0.0
        return
    end

    θ_not_ode = @view θ_est[θ_indices.iθ_not_ode]
    ReverseDiff.gradient!(petab_ODE_cache.gradient_θ_not_ode, compute_cost_θ_not_ODE, θ_not_ode)
    @views gradient[θ_indices.iθ_not_ode] .= petab_ODE_cache.gradient_θ_not_ode

    if prior_info.has_priors == true
        PEtab.compute_gradient_prior!(gradient, θ_est, θ_indices, prior_info)
    end
end


# Compute the adjoint gradient across all experimental conditions
function compute_gradient_adjoint_θ_dynamic!(gradient::Vector{Float64},
                                             θ_dynamic::Vector{Float64},
                                             θ_sd::Vector{Float64},
                                             θ_observable::Vector{Float64},
                                             θ_non_dynamic::Vector{Float64},
                                             ode_problem::ODEProblem,
                                             ode_solver::ODESolver,
                                             ss_solver::SteadyStateSolver,
                                             sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                             petab_model::PEtabModel,
                                             simulation_info::PEtab.SimulationInfo,
                                             θ_indices::PEtab.ParameterIndices,
                                             measurement_info::PEtab.MeasurementsInfo,
                                             parameter_info::PEtab.ParametersInfo,
                                             petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                             petab_ODESolver_cache::PEtab.PEtabODESolverCache;
                                             sensealg_ss=SteadyStateAdjoint(),
                                             exp_id_solve::Vector{Symbol} = [:all])

    θ_dynamicT = PEtab.transformθ(θ_dynamic, θ_indices.θ_dynamic_names, θ_indices, :θ_dynamic, petab_ODE_cache)
    θ_sdT = PEtab.transformθ(θ_sd, θ_indices.θ_sd_names, θ_indices, :θ_sd, petab_ODE_cache)
    θ_observableT = PEtab.transformθ(θ_observable, θ_indices.θ_observable_names, θ_indices, :θ_observable, petab_ODE_cache)
    θ_non_dynamicT = PEtab.transformθ(θ_non_dynamic, θ_indices.θ_non_dynamic_names, θ_indices, :θ_non_dynamic, petab_ODE_cache)

    _ode_problem = remake(ode_problem, p = convert.(eltype(θ_dynamicT), ode_problem.p), u0 = convert.(eltype(θ_dynamicT), ode_problem.u0))
    PEtab.change_ode_parameters!(_ode_problem.p, _ode_problem.u0, θ_dynamicT, θ_indices, petab_model)
    success = PEtab.solve_ode_all_conditions!(simulation_info.ode_sols_derivatives, _ode_problem, petab_model, θ_dynamicT, petab_ODESolver_cache, simulation_info, θ_indices, ode_solver, ss_solver, exp_id_solve=exp_id_solve, dense_sol=true, save_at_observed_t=false, track_callback=true)
    if success != true
        gradient .= 1e8
        return
    end

    # In case of PreEq-critera we need to compute the pullback function at tSS to compute the VJP between
    # λ_t0 and the sensitivites at steady state time
    if simulation_info.has_pre_equilibration_condition_id == true
        _eval_VJP_ss = generate_VJP_ss(simulation_info, sensealg_ss, ode_solver, ss_solver, exp_id_solve)
    end

    fill!(gradient, 0.0)
    # Compute the gradient by looping through all experimental conditions.
    for i in eachindex(simulation_info.experimental_condition_id)
        experimental_condition_id = simulation_info.experimental_condition_id[i]
        simulation_condition_id = simulation_info.simulation_condition_id[i]

        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        if simulation_info.has_pre_equilibration_condition_id == true
            eval_VJP_ss = _eval_VJP_ss[simulation_info.pre_equilibration_condition_id[i]]
        else
            eval_VJP_ss = identity
        end

        # In case the model is simulated first to a steady state we need to keep track of the post-equlibrium experimental
        # condition Id to identify parameters specific to an experimental condition.
        sol = simulation_info.ode_sols_derivatives[experimental_condition_id]
        success = compute_gradient_adjoint_condition!(gradient, sol, petab_ODE_cache, sensealg, ode_solver,
                                                      θ_dynamicT, θ_sdT, θ_observableT,  θ_non_dynamicT, 
                                                      experimental_condition_id, simulation_condition_id, simulation_info,
                                                      petab_model, θ_indices, measurement_info, parameter_info, eval_VJP_ss)

        if success == false
            fill!(gradient, 0.0)
            return
        end
    end
    return
end


function generate_VJP_ss(simulation_info::PEtab.SimulationInfo,
                         sensealg_ss::SteadyStateAdjoint,
                         ode_solver::ODESolver,
                         ss_solver::SteadyStateSolver,
                         exp_id_solve::Vector{Symbol})::NamedTuple

    # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
    # (exp_id_solve != [["all]]) the number of preEq cond. might be smaller than the
    # total number of preEq cond.
    if exp_id_solve[1] == :all
        pre_equilibration_condition_id = unique(simulation_info.pre_equilibration_condition_id)
    else
        which_ids  = findall(x -> x ∈ simulation_info.experimental_condition_id, exp_id_solve)
        pre_equilibration_condition_id = unique(simulation_info.pre_equilibration_condition_id[which_ids])
    end

    _eval_VJP_ss = Vector{Function}(undef, length(pre_equilibration_condition_id))
    solver, abstol, reltol, force_dtmin, dtmin, maxiters = ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ode_solver.force_dtmin, ode_solver.dtmin, ode_solver.maxiters
    for i in eachindex(pre_equilibration_condition_id)

        ode_problem = simulation_info.ode_sols_pre_equlibrium[pre_equilibration_condition_id[i]].prob
        ssOdeProblem = SteadyStateProblem(ode_problem)
        ySS, _eval_VJP_ss_i = Zygote.pullback((p) ->    (
                                                      solve(ssOdeProblem,
                                                            SteadyStateDiffEq.DynamicSS(solver, abstol=ss_solver.abstol, reltol=ss_solver.reltol),
                                                            abstol=abstol,
                                                            reltol=reltol,
                                                            maxiters=maxiters,
                                                            force_dtmin=force_dtmin,
                                                            p=p,
                                                            sensealg=sensealg_ss)[:]), ode_problem.p)

        _eval_VJP_ss[i] = (du) -> begin return _eval_VJP_ss_i(du)[1] end
    end

    eval_VJP_ss = Tuple(f for f in _eval_VJP_ss)
    return NamedTuple{Tuple(name for name in pre_equilibration_condition_id)}(eval_VJP_ss)
end
function generate_VJP_ss(simulation_info::PEtab.SimulationInfo,
                               sensealg_ss::Union{QuadratureAdjoint, InterpolatingAdjoint},
                               ode_solver::ODESolver,
                               ss_solver::SteadyStateSolver,
                               exp_id_solve::Vector{Symbol})::NamedTuple

    # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
    # (exp_id_solve != [["all]]) the number of preEq cond. might be smaller than the
    # total number of preEq cond.
    if exp_id_solve[1] == :all
        pre_equilibration_condition_id = unique(simulation_info.pre_equilibration_condition_id)
    else
        which_ids  = findall(x -> x ∈ simulation_info.experimental_condition_id, exp_id_solve)
        pre_equilibration_condition_id = unique(simulation_info.pre_equilibration_condition_id[which_ids])
    end

    _eval_VJP_ss = Vector{Function}(undef, length(pre_equilibration_condition_id))
    solver, abstol, reltol, force_dtmin, dtmin, maxiters = ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ode_solver.force_dtmin, ode_solver.dtmin, ode_solver.maxiters
    for i in eachindex(pre_equilibration_condition_id)

        # Sets up a function which takes du and solves the Adjoint ODE system with du
        # as starting point. This is a temporary ugly solution as there are some problems
        # with retcode Terminated and using CVODE_BDF
        _sol = simulation_info.ode_sols_pre_equlibrium[pre_equilibration_condition_id[i]]
        _prob = remake(_sol.prob, tspan=(0.0, _sol.t[end]))
        sol = solve(_prob, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, sensealg_ss, maxiters=maxiters)

        _eval_VJP_ss_i = (du) -> compute_VJP_ss(du, sol, solver, sensealg_ss, reltol, abstol, dtmin, force_dtmin, maxiters)
        _eval_VJP_ss[i] = _eval_VJP_ss_i
    end

    eval_VJP_ss = Tuple(f for f in _eval_VJP_ss)
    return NamedTuple{Tuple(name for name in pre_equilibration_condition_id)}(eval_VJP_ss)
end


# Compute the adjoint VJP for steady state simulated models via QuadratureAdjoint and InterpolatingAdjoint
# by, given du as initial values, solve the adjoint integral.
# TODO : Add interface for SteadyStateAdjoint
function compute_VJP_ss(du::AbstractVector,
                        _sol::ODESolution,
                        odeSolver::SciMLAlgorithm,
                        sensealg::QuadratureAdjoint,
                        reltol::Float64,
                        abstol::Float64,
                        dtmin::Union{Float64, Nothing},
                        force_dtmin::Bool,
                        maxiters::Int64)

    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, odeSolver, [_sol.t[end]], compute_∂g∂u_empty, nothing,
                                      nothing, nothing, nothing, Val(true))
    adj_prob.u0 .= du
    adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = true, save_start = true)
    integrand = AdjointSensitivityIntegrand(_sol, adj_sol, sensealg, nothing)
    res, err = SciMLSensitivity.quadgk(integrand, _sol.prob.tspan[1], _sol.t[end],
                                       atol = abstol, rtol = reltol)
    return res'
end
function compute_VJP_ss(du::AbstractVector,
                        _sol::ODESolution,
                        odeSolver::SciMLAlgorithm,
                        sensealg::InterpolatingAdjoint,
                        reltol::Float64,
                        abstol::Float64,
                        dtmin::Union{Float64, Nothing},
                        force_dtmin::Bool,
                        maxiters::Int64)

    n_model_states = length(_sol.prob.u0)
    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, odeSolver, [_sol.t[end]], compute_∂g∂u_empty, nothing,
                                      nothing, nothing, nothing, Val(true))

    adj_prob.u0[1:n_model_states] .= du[1:n_model_states]

    adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = true, save_start = true)
    out = adj_sol[end][(n_model_states+1):end]
    return out
end


function compute_∂g∂u_empty(out, u, p, t, i)
    out .= 0.0
end


# For a given experimental condition compute the gradient using adjoint sensitivity analysis
# for a funciton on the form G = (h - y_obs)^2 / σ^2
# TODO : Important function - improve documentation.
function compute_gradient_adjoint_condition!(gradient::Vector{Float64},
                                             sol::ODESolution,
                                             petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                             sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                             ode_solver::ODESolver,
                                             θ_dynamic::Vector{Float64},
                                             θ_sd::Vector{Float64},
                                             θ_observable::Vector{Float64},
                                             θ_non_dynamic::Vector{Float64},
                                             experimental_condition_id::Symbol,
                                             simulation_condition_id::Symbol,
                                             simulation_info::PEtab.SimulationInfo,
                                             petab_model::PEtab.PEtabModel,
                                             θ_indices::PEtab.ParameterIndices,
                                             measurement_info::PEtab.MeasurementsInfo,
                                             parameter_info::PEtab.ParametersInfo,
                                             eval_VJP_ss::Function)::Bool

    # Extract experimetnalCondition specific parameter required to solve the
    # adjoitn ODE
    i_per_time_point = simulation_info.i_per_time_point[experimental_condition_id]
    time_observed = simulation_info.time_observed[experimental_condition_id]
    callback = simulation_info.tracked_callbacks[experimental_condition_id]

    compute∂G∂u! = (out, u, p, t, i) -> begin PEtab.compute∂G∂_(out, u, p, t, i, i_per_time_point,
                                                                measurement_info, parameter_info,
                                                                θ_indices, petab_model,
                                                                θ_sd, θ_observable,  θ_non_dynamic,
                                                                petab_ODE_cache.∂h∂u, petab_ODE_cache.∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p! = (out, u, p, t, i) -> begin PEtab.compute∂G∂_(out, u, p, t, i, i_per_time_point,
                                                                measurement_info, parameter_info,
                                                                θ_indices, petab_model,
                                                                θ_sd, θ_observable,  θ_non_dynamic,
                                                                petab_ODE_cache.∂h∂p, petab_ODE_cache.∂σ∂p, compute∂G∂U=false)
                                        end

    solver, abstol, reltol, force_dtmin, dtmin, maxiters = ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ode_solver.force_dtmin, ode_solver.dtmin, ode_solver.maxiters

    # The standard allow cases where we only observe data at t0, that is we do not solve the ODE. Here adjoint_sensitivities fails (naturally). In this case we compute the gradient
    # via ∇G_p = dp + du*J(u(t_0)) where du is the cost function differentiated with respect to the states at time zero,
    # dp is the cost function  with respect to the parameters at time zero and J is sensititvites at time
    # zero. Overall, the only workflow that changes below is that we compute du outside of the adjoint interface
    # and use sol[:] as we no longer can interpolate from the forward solution.
    only_obs_at_zero::Bool = false
    du = petab_ODE_cache.du
    dp = petab_ODE_cache.dp
    if !(length(time_observed) == 1 && time_observed[1] == 0.0)

        status = __adjoint_sensitivities!(du, dp, sol, sensealg, time_observed, solver, abstol, reltol,
                                          force_dtmin, dtmin, maxiters, callback, compute∂G∂u!)
        status == false && return false
    else
        compute∂G∂u!(du, sol[1], sol.prob.p, 0.0, 1)
        only_obs_at_zero = true
    end
    # Technically we can pass compute∂G∂p above to dgdp_discrete. However, odeProb.p often contain
    # constant parameters which are not a part ode the parameter estimation problem. Sometimes
    # the gradient for these evaluate to NaN (as they where never thought to be estimated) which
    # results in the entire gradient evaluating to NaN. Hence, we perform this calculation outside
    # of the lower level interface.
    ∂G∂p, ∂G∂p_ = petab_ODE_cache.∂G∂p_, petab_ODE_cache.∂G∂p
    fill!(∂G∂p, 0.0)
    for i in eachindex(time_observed)
        if only_obs_at_zero == false
            compute∂G∂p!(∂G∂p_, sol(time_observed[i]), sol.prob.p, time_observed[i], i)
        else
            compute∂G∂p!(∂G∂p_, sol[1], sol.prob.p, time_observed[i], i)
        end
        ∂G∂p .+= ∂G∂p_
    end

    _gradient = petab_ODE_cache._gradient_adjoint
    if simulation_info.has_pre_equilibration_condition_id == false
        # In case we do not simulate the ODE for a steady state first we can compute
        # the initial sensitivites easily via automatic differantitatiom
        S_t0 = petab_ODE_cache.S_t0
        ForwardDiff.jacobian!(S_t0, petab_model.compute_u0!, sol.prob.u0, sol.prob.p)
        _gradient .= dp .+ transpose(S_t0) * du

    else
        # In case we simulate to a stady state we need to compute a VJP. We use
        # Zygote pullback to avoid having to having build the Jacobian, rather
        # we create the yBar function required for the vector Jacobian product.
        @views _gradient .= dp .+ eval_VJP_ss(du)
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    PEtab.adjust_gradient_θ_Transformed!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices,
                                               simulation_condition_id, adjoint=true)
    return true
end


# In order to obtain the ret-codes when solving the adjoint ODE system we must, as here copy to 99% from SciMLSensitivity
# GitHub repo to access the actual solve-call to the ODEAdjointProblem
function __adjoint_sensitivities!(_du::AbstractVector,
                                  _dp::AbstractVector,
                                  sol::ODESolution,
                                  sensealg::InterpolatingAdjoint,
                                  t::Vector{Float64},
                                  odeSolver::SciMLAlgorithm,
                                  abstol::Float64,
                                  reltol::Float64,
                                  force_dtmin::Bool,
                                  dtmin::Union{Float64, Nothing},
                                  maxiters::Int64,
                                  callback::SciMLBase.DECallback,
                                  compute_∂G∂u::F)::Bool where F

    rcb = nothing
    adjProb, rcb = ODEAdjointProblem(sol, sensealg, odeSolver, t,
                                     compute_∂G∂u, nothing, nothing, nothing, nothing, Val(true);
                                     abstol=abstol, reltol=reltol, callback=callback)

    tstops = SciMLSensitivity.ischeckpointing(sensealg, sol) ? checkpoints : similar(sol.t, 0)
    if isnothing(dtmin)
        adj_sol = solve(adjProb, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = false, save_start = false, saveat = eltype(sol[1])[],
                    tstops=tstops)
    else
        adj_sol = solve(adjProb, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = false, save_start = false, saveat = eltype(sol[1])[],
                    tstops=tstops, dtmin=dtmin)
    end
    if adj_sol.retcode != ReturnCode.Success
        _du .= 0.0
        _dp .= 0.0
        return false
    end

    p = sol.prob.p
    l = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(sol.prob.p)
    du0 = adj_sol[end][1:length(sol.prob.u0)]

    if eltype(sol.prob.p) <: real(eltype(adj_sol[end]))
        dp = real.(adj_sol[end][(1:l) .+ length(sol.prob.u0)])'
    elseif p === nothing || p === DiffEqBase.NullParameters()
        dp = nothing
    else
        dp = adj_sol[end][(1:l) .+ length(sol.prob.u0)]'
    end

    if rcb !== nothing && !isempty(rcb.Δλas)
        S = adj_prob.f.f
        iλ = similar(rcb.λ, length(first(sol.u)))
        out = zero(dp')
        yy = similar(rcb.y)
        for (Δλa, tt) in rcb.Δλas
            iλ .= zero(eltype(iλ))
            @unpack algevar_idxs = rcb.diffcache
            iλ[algevar_idxs] .= Δλa
            sol(yy, tt)
            vecjacobian!(nothing, yy, iλ, sol.prob.p, tt, S, dgrad = out)
            dp .+= out'
        end
    end

    _du .= du0
    _dp .= dp'
    return true
end
function __adjoint_sensitivities!(_du::AbstractVector,
                                  _dp::AbstractVector,
                                  sol::ODESolution,
                                  sensealg::QuadratureAdjoint,
                                  t::Vector{Float64},
                                  odeSolver::SciMLAlgorithm,
                                  abstol::Float64,
                                  reltol::Float64,
                                  force_dtmin::Bool,
                                  dtmin::Union{Float64, Nothing},
                                  maxiters::Int64,
                                  callback::SciMLBase.DECallback,
                                  compute_∂G∂u::F)::Bool where F

    adj_prob, rcb = ODEAdjointProblem(sol, sensealg, odeSolver, t, compute_∂G∂u, nothing,
                                      nothing, nothing, nothing, Val(true);
                                      callback)
    if isnothing(dtmin)                                      
        adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                        save_everystep = true, save_start = true)
    else
        adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                        save_everystep = true, save_start = true, dtmin=dtmin)
    end

    if adj_sol.retcode != ReturnCode.Success
        _du .= 0.0
        _dp .= 0.0
        return false
    end

    p = sol.prob.p
    if p === nothing || p === DiffEqBase.NullParameters()
        _du .= adj_sol[end]
        return true
    else
        integrand = AdjointSensitivityIntegrand(sol, adj_sol, sensealg, nothing)
        if t === nothing
            res, err = SciMLSensitivity.quadgk(integrand, sol.prob.tspan[1], sol.prob.tspan[2],
                              atol = abstol, rtol = reltol)
        else
            res = zero(integrand.p)'

            if callback !== nothing
                cur_time = length(t)
                dλ = similar(integrand.λ)
                dλ .*= false
                dgrad = similar(res)
                dgrad .*= false
            end

            # correction for end interval.
            if t[end] != sol.prob.tspan[2] && sol.retcode !== ReturnCode.Terminated
                res .+= SciMLSensitivity.quadgk(integrand, t[end], sol.prob.tspan[end],
                               atol = abstol, rtol = reltol)[1]
            end

            if sol.retcode === ReturnCode.Terminated
                integrand = update_integrand_and_dgrad(res, sensealg, callback, integrand,
                                                       adj_prob, sol, compute_∂G∂u,
                                                       nothing, dλ, dgrad, t[end],
                                                       cur_time)
            end

            for i in (length(t) - 1):-1:1
                res .+= SciMLSensitivity.quadgk(integrand, t[i], t[i + 1],
                                   atol = abstol, rtol = reltol)[1]
                if t[i] == t[i + 1]
                    integrand = SciMLSensitivity.update_integrand_and_dgrad(res, sensealg, callback,
                                                                            integrand,
                                                                            adj_prob, sol, compute_∂G∂u,
                                                                            nothing, dλ, dgrad, t[i],
                                                                            cur_time)
                end
                (callback !== nothing || dgdp_discrete !== nothing) &&
                    (cur_time -= one(cur_time))
            end
            # correction for start interval
            if t[1] != sol.prob.tspan[1]
                res .+= SciMLSensitivity.quadgk(integrand, sol.prob.tspan[1], t[1],
                               atol = abstol, rtol = reltol)[1]
            end
        end
    end

    if rcb !== nothing && !isempty(rcb.Δλas)
        iλ = zero(rcb.λ)
        out = zero(res')
        yy = similar(rcb.y)
        for (Δλa, tt) in rcb.Δλas
            @unpack algevar_idxs = rcb.diffcache
            iλ[algevar_idxs] .= Δλa
            sol(yy, tt)
            vec_pjac!(out, iλ, yy, tt, integrand)
            res .+= out'
            iλ .= zero(eltype(iλ))
        end
    end

    _du .= adj_sol[end]
    _dp .= res'
    return true
end
