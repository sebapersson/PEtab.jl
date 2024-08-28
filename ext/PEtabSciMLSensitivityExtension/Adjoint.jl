# Compute gradient via adjoint sensitivity analysis
function compute_gradient_adjoint!(gradient::Vector{Float64},
                                   θ_est::Vector{Float64},
                                   compute_cost_θ_not_ODE::Function,
                                   probleminfo::PEtab.PEtabODEProblemInfo,
                                   model_info::PEtab.ModelInfo;
                                   exp_id_solve::Vector{Symbol} = [:all])::Nothing
    @unpack cache, sensealg, sensealg_ss = probleminfo
    @unpack simulation_info, petab_model, simulation_info, θ_indices = model_info
    @unpack measurement_info, parameter_info, prior_info = model_info
    ss_solver = probleminfo.ss_solver_gradient
    ode_solver = probleminfo.solver_gradient
    ode_problem = probleminfo.odeproblem_gradient

    PEtab.splitθ!(θ_est, θ_indices, cache)
    xdynamic = cache.xdynamic
    xobservable = cache.xobservable
    xnoise = cache.xnoise
    xnondynamic = cache.xnondynamic

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    compute_gradient_adjoint_xdynamic!(cache.xdynamic_grad, xdynamic, xnoise, xobservable,
                                       xnondynamic, model_info, probleminfo, ode_problem,
                                       ode_solver, ss_solver, sensealg, petab_model,
                                       simulation_info, θ_indices, measurement_info,
                                       parameter_info, cache; exp_id_solve = exp_id_solve,
                                       sensealg_ss = sensealg_ss)
    @views gradient[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if !isempty(cache.xdynamic_grad) &&
       all(cache.xdynamic_grad .== 0.0)
        gradient .= 0.0
        return nothing
    end

    θ_not_ode = @view θ_est[θ_indices.xindices[:not_system]]
    ReverseDiff.gradient!(cache.xnotode_grad, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.xindices[:not_system]] .= cache.xnotode_grad

    if prior_info.has_priors == true
        PEtab.grad_prior!(gradient, θ_est, θ_indices, prior_info)
    end

    return nothing
end

# Compute the adjoint gradient across all experimental conditions
function compute_gradient_adjoint_xdynamic!(gradient::Vector{Float64},
                                             xdynamic::Vector{Float64},
                                             xnoise::Vector{Float64},
                                             xobservable::Vector{Float64},
                                             xnondynamic::Vector{Float64},
                                             model_info::PEtab.ModelInfo,
                                             probleminfo::PEtab.PEtabODEProblemInfo,
                                             ode_problem::ODEProblem,
                                             ode_solver::ODESolver,
                                             ss_solver::SteadyStateSolver,
                                             sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                             petab_model::PEtabModel,
                                             simulation_info::PEtab.SimulationInfo,
                                             θ_indices::PEtab.ParameterIndices,
                                             measurement_info::PEtab.MeasurementsInfo,
                                             parameter_info::PEtab.ParametersInfo,
                                             cache::PEtab.PEtabODEProblemCache;
                                             sensealg_ss = SteadyStateAdjoint(),
                                             exp_id_solve::Vector{Symbol} = [:all])::Nothing
    xdynamic_ps = PEtab.transform_x(xdynamic, θ_indices.xids[:dynamic], θ_indices,
                                  :xdynamic, cache)
    xnoise_ps = PEtab.transform_x(xnoise, θ_indices.xids[:noise], θ_indices, :xnoise,
                             cache)
    xobservable_ps = PEtab.transform_x(xobservable, θ_indices.xids[:observable], θ_indices,
                                     :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(xnondynamic, θ_indices.xids[:nondynamic],
                                      θ_indices, :xnondynamic, cache)

    _ode_problem = remake(ode_problem, p = convert.(eltype(xdynamic_ps), ode_problem.p),
                          u0 = convert.(eltype(xdynamic_ps), ode_problem.u0))
    PEtab.change_ode_parameters!(_ode_problem.p, _ode_problem.u0, xdynamic_ps, θ_indices,
                                 petab_model)
    success = PEtab.solve_ode_all_conditions!(model_info, xdynamic_ps, probleminfo;
                                              exp_id_solve = exp_id_solve, dense_sol = true,
                                              save_at_observed_t = false,
                                              track_callback = true)
    if success != true
        gradient .= 1e8
        return nothing
    end

    # In case of PreEq-critera we need to compute the pullback function at tSS to compute the VJP between
    # λ_t0 and the sensitivites at steady state time
    if simulation_info.has_pre_equilibration == true
        _eval_VJP_ss = generate_VJP_ss(simulation_info, sensealg_ss, ode_solver, ss_solver,
                                       exp_id_solve)
    end

    fill!(gradient, 0.0)
    # Compute the gradient by looping through all experimental conditions.
    for i in eachindex(simulation_info.conditionids[:experiment])
        experimental_condition_id = simulation_info.conditionids[:experiment][i]
        simulation_condition_id = simulation_info.conditionids[:simulation][i]

        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        if simulation_info.has_pre_equilibration == true
            eval_VJP_ss = _eval_VJP_ss[simulation_info.conditionids[:pre_equilibration][i]]
        else
            eval_VJP_ss = identity
        end

        # In case the model is simulated first to a steady state we need to keep track of the post-equlibrium experimental
        # condition Id to identify parameters specific to an experimental condition.
        sol = simulation_info.odesols_derivatives[experimental_condition_id]
        success = compute_gradient_adjoint_condition!(gradient, sol, cache,
                                                      sensealg, ode_solver,
                                                      xdynamic_ps, xnoise_ps, xobservable_ps,
                                                      xnondynamic_ps,
                                                      experimental_condition_id,
                                                      simulation_condition_id,
                                                      simulation_info,
                                                      petab_model, θ_indices,
                                                      measurement_info, parameter_info,
                                                      eval_VJP_ss)

        if success == false
            fill!(gradient, 0.0)
            return nothing
        end
    end
    return nothing
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
        pre_equilibration_condition_id = unique(simulation_info.conditionids[:pre_equilibration])
    else
        which_ids = findall(x -> x ∈ simulation_info.conditionids[:experiment],
                            exp_id_solve)
        pre_equilibration_condition_id = unique(simulation_info.conditionids[:pre_equilibration][which_ids])
    end

    _eval_VJP_ss = Vector{Function}(undef, length(pre_equilibration_condition_id))
    solver, abstol, reltol, force_dtmin, dtmin, maxiters = ode_solver.solver,
                                                           ode_solver.abstol,
                                                           ode_solver.reltol,
                                                           ode_solver.force_dtmin,
                                                           ode_solver.dtmin,
                                                           ode_solver.maxiters
    for i in eachindex(pre_equilibration_condition_id)
        ode_problem = simulation_info.odesols_preeq[pre_equilibration_condition_id[i]].prob
        ssOdeProblem = SteadyStateProblem(ode_problem)
        ySS, _eval_VJP_ss_i = Zygote.pullback((p) -> (solve(ssOdeProblem,
                                                            SteadyStateDiffEq.DynamicSS(solver),
                                                            abstol = ss_solver.abstol,
                                                            reltol = ss_solver.reltol,
                                                            odesolve_kwargs = (abstol = abstol,
                                                                               reltol = reltol),
                                                            maxiters = maxiters,
                                                            force_dtmin = force_dtmin,
                                                            p = p,
                                                            sensealg = sensealg_ss)[:]),
                                              ode_problem.p)

        _eval_VJP_ss[i] = (du) -> begin
            return _eval_VJP_ss_i(du)[1]
        end
    end

    eval_VJP_ss = Tuple(f for f in _eval_VJP_ss)
    return NamedTuple{Tuple(name for name in pre_equilibration_condition_id)}(eval_VJP_ss)
end
function generate_VJP_ss(simulation_info::PEtab.SimulationInfo,
                         sensealg_ss::Union{QuadratureAdjoint, InterpolatingAdjoint,
                                            GaussAdjoint},
                         ode_solver::ODESolver,
                         ss_solver::SteadyStateSolver,
                         exp_id_solve::Vector{Symbol})::NamedTuple

    # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
    # (exp_id_solve != [["all]]) the number of preEq cond. might be smaller than the
    # total number of preEq cond.
    if exp_id_solve[1] == :all
        pre_equilibration_condition_id = unique(simulation_info.conditionids[:pre_equilibration])
    else
        which_ids = findall(x -> x ∈ simulation_info.conditionids[:experiment],
                            exp_id_solve)
        pre_equilibration_condition_id = unique(simulation_info.conditionids[:pre_equilibration][which_ids])
    end

    _eval_VJP_ss = Vector{Function}(undef, length(pre_equilibration_condition_id))
    @unpack solver, abstol, reltol, force_dtmin, dtmin, maxiters = ode_solver
    for i in eachindex(pre_equilibration_condition_id)

        # Sets up a function which takes du and solves the Adjoint ODE system with du
        # as starting point. This is a temporary ugly solution as there are some problems
        # with retcode Terminated and using CVODE_BDF
        _sol = simulation_info.odesols_preeq[pre_equilibration_condition_id[i]]
        _prob = remake(_sol.prob, tspan = (0.0, _sol.t[end]))
        sol = solve(_prob, solver, abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters)

        _eval_VJP_ss_i = (du) -> compute_VJP_ss(du, sol, solver, sensealg_ss, reltol,
                                                abstol, dtmin, force_dtmin, maxiters)
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
                        solver::SciMLAlgorithm,
                        sensealg::QuadratureAdjoint,
                        reltol::Float64,
                        abstol::Float64,
                        dtmin::Union{Float64, Nothing},
                        force_dtmin::Bool,
                        maxiters::Int64)::AbstractVector
    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, solver, [_sol.t[end]],
                                      compute_∂g∂u_empty, nothing, nothing, nothing,
                                      nothing, Val(true))
    adj_prob.u0 .= du
    adj_sol = solve(adj_prob, solver; abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters, save_everystep = true,
                    save_start = true)
    integrand = AdjointSensitivityIntegrand(_sol, adj_sol, sensealg, nothing)
    res, err = SciMLSensitivity.quadgk(integrand, _sol.prob.tspan[1], _sol.t[end],
                                       atol = abstol, rtol = reltol)
    return res'
end
function compute_VJP_ss(du::AbstractVector,
                        _sol::ODESolution,
                        solver::SciMLAlgorithm,
                        sensealg::InterpolatingAdjoint,
                        reltol::Float64,
                        abstol::Float64,
                        dtmin::Union{Float64, Nothing},
                        force_dtmin::Bool,
                        maxiters::Int64)::AbstractVector
    n_model_states = length(_sol.prob.u0)
    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, solver, [_sol.t[end]],
                                      compute_∂g∂u_empty, nothing, nothing, nothing,
                                      nothing, Val(true))

    adj_prob.u0[1:n_model_states] .= du[1:n_model_states]

    adj_sol = solve(adj_prob, solver; abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters,
                    save_everystep = true, save_start = true)
    out = adj_sol[end][(n_model_states + 1):end]
    return out
end

function compute_∂g∂u_empty(out, u, p, t, i)::Nothing
    out .= 0.0
    return nothing
end

# For a given experimental condition compute the gradient using adjoint sensitivity analysis
# for a funciton on the form G = (h - y_obs)^2 / σ^2
# TODO : Important function - improve documentation.
function compute_gradient_adjoint_condition!(gradient::Vector{Float64},
                                             sol::ODESolution,
                                             cache::PEtab.PEtabODEProblemCache,
                                             sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                             ode_solver::ODESolver,
                                             xdynamic::Vector{Float64},
                                             xnoise::Vector{Float64},
                                             xobservable::Vector{Float64},
                                             xnondynamic::Vector{Float64},
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
    imeasurements_t = simulation_info.imeasurements_t[experimental_condition_id]
    time_observed = simulation_info.tsaves[experimental_condition_id]
    callback = simulation_info.tracked_callbacks[experimental_condition_id]

    compute∂G∂u! = (out, u, p, t, i) -> begin
        PEtab.compute∂G∂_(out, u, p, t, i, imeasurements_t,
                          measurement_info, parameter_info,
                          θ_indices, petab_model,
                          xnoise, xobservable, xnondynamic,
                          cache.∂h∂u, cache.∂σ∂u, compute∂G∂U = true)
    end
    compute∂G∂p! = (out, u, p, t, i) -> begin
        PEtab.compute∂G∂_(out, u, p, t, i, imeasurements_t,
                          measurement_info, parameter_info,
                          θ_indices, petab_model,
                          xnoise, xobservable, xnondynamic,
                          cache.∂h∂p, cache.∂σ∂p, compute∂G∂U = false)
    end

    @unpack solver_adj, abstol_adj, reltol_adj, maxiters, force_dtmin = ode_solver

    # The standard allow cases where we only observe data at t0, that is we do not solve the ODE. Here adjoint_sensitivities fails (naturally). In this case we compute the gradient
    # via ∇G_p = dp + du*J(u(t_0)) where du is the cost function differentiated with respect to the states at time zero,
    # dp is the cost function  with respect to the parameters at time zero and J is sensititvites at time
    # zero. Overall, the only workflow that changes below is that we compute du outside of the adjoint interface
    # and use sol[:] as we no longer can interpolate from the forward solution.
    only_obs_at_zero::Bool = false
    du = cache.du
    dp = cache.dp
    if !(length(time_observed) == 1 && time_observed[1] == 0.0)
        status = __adjoint_sensitivities!(du, dp, sol, sensealg, time_observed, solver_adj,
                                          abstol_adj, reltol_adj, callback, compute∂G∂u!;
                                          maxiters = maxiters, force_dtmin = force_dtmin)
        status == false && return false
    else
        compute∂G∂u!(du, sol[1], sol.prob.p, 0.0, 1)
        only_obs_at_zero = true
    end
    # Technically we can pass compute∂G∂p above to dgdp_discrete. However, ode_problem.p often contain
    # constant parameters which are not a part ode the parameter estimation problem. Sometimes
    # the gradient for these evaluate to NaN (as they where never thought to be estimated) which
    # results in the entire gradient evaluating to NaN. Hence, we perform this calculation outside
    # of the lower level interface.
    ∂G∂p, ∂G∂p_ = cache.∂G∂p_, cache.∂G∂p
    fill!(∂G∂p, 0.0)
    for i in eachindex(time_observed)
        if only_obs_at_zero == false
            compute∂G∂p!(∂G∂p_, sol(time_observed[i]), sol.prob.p, time_observed[i], i)
        else
            compute∂G∂p!(∂G∂p_, sol[1], sol.prob.p, time_observed[i], i)
        end
        ∂G∂p .+= ∂G∂p_
    end

    _gradient = cache.adjoint_grad
    if simulation_info.has_pre_equilibration == false
        # In case we do not simulate the ODE for a steady state first we can compute
        # the initial sensitivites easily via automatic differantitatiom
        St0 = cache.St0
        ForwardDiff.jacobian!(St0, petab_model.compute_u0!, sol.prob.u0, sol.prob.p)
        _gradient .= dp .+ transpose(St0) * du

    else
        # In case we simulate to a stady state we need to compute a VJP. We use
        # Zygote pullback to avoid having to having build the Jacobian, rather
        # we create the yBar function required for the vector Jacobian product.
        @views _gradient .= dp .+ eval_VJP_ss(du)
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log.
    PEtab.adjust_gradient_θ_transformed!(gradient, _gradient, ∂G∂p, xdynamic, θ_indices,
                                         simulation_condition_id, adjoint = true)
    return true
end

# In order to obtain the ret-codes when solving the adjoint ODE system we must, as here copy to 99% from SciMLSensitivity
# GitHub repo to access the actual solve-call to the ODEAdjointProblem
function __adjoint_sensitivities!(_du::AbstractVector,
                                  _dp::AbstractVector,
                                  sol::ODESolution,
                                  sensealg::InterpolatingAdjoint,
                                  t::Vector{Float64},
                                  solver::SciMLAlgorithm,
                                  abstol::Float64,
                                  reltol::Float64,
                                  callback::SciMLBase.DECallback,
                                  compute_∂G∂u::F;
                                  kwargs...)::Bool where {F}
    rcb = nothing
    adj_prob, rcb = ODEAdjointProblem(sol, sensealg, solver, t, compute_∂G∂u,
                                      nothing, nothing, nothing, nothing, Val(true);
                                      abstol = abstol, reltol = reltol, callback = callback)

    tstops = SciMLSensitivity.ischeckpointing(sensealg, sol) ? checkpoints :
             similar(sol.t, 0)
    adj_sol = solve(adj_prob, solver; save_everystep = false, save_start = false,
                    saveat = eltype(sol.u[1])[], tstops = tstops, abstol = abstol,
                    reltol = reltol)

    if adj_sol.retcode != ReturnCode.Success
        _du .= 0.0
        _dp .= 0.0
        return false
    end

    p = sol.prob.p
    l = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(sol.prob.p)
    du0 = adj_sol.u[end][1:length(sol.prob.u0)]

    if eltype(sol.prob.p) <: real(eltype(adj_sol.u[end]))
        dp = real.(adj_sol.u[end][(1:l) .+ length(sol.prob.u0)])'
    elseif p === nothing || p === DiffEqBase.NullParameters()
        dp = nothing
    else
        dp = adj_sol.u[end][(1:l) .+ length(sol.prob.u0)]'
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
                                  solver::SciMLAlgorithm,
                                  abstol::Float64,
                                  reltol::Float64,
                                  callback::SciMLBase.DECallback,
                                  compute_∂G∂u::F;
                                  kwargs...)::Bool where {F}
    adj_prob, rcb = ODEAdjointProblem(sol, sensealg, solver, t, compute_∂G∂u, nothing,
                                      nothing, nothing, nothing, Val(true); callback)
    adj_sol = solve(adj_prob, solver; abstol = abstol, reltol = reltol,
                    save_everystep = true, save_start = true, kwargs...)

    if adj_sol.retcode != ReturnCode.Success
        _du .= 0.0
        _dp .= 0.0
        return false
    end

    p = sol.prob.p
    if p === nothing || p === DiffEqBase.NullParameters()
        _du .= adj_sol[end]
        _dp .= 0.0
    else
        integrand = AdjointSensitivityIntegrand(sol, adj_sol, sensealg, nothing)
        if t === nothing
            res, err = SciMLSensitivity.quadgk(integrand, sol.prob.tspan[1],
                                               sol.prob.tspan[2],
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
                if ArrayInterface.ismutable(res)
                    res .+= SciMLSensitivity.quadgk(integrand, t[i], t[i + 1],
                                                    atol = abstol, rtol = reltol)[1]
                else
                    res += SciMLSensitivity.quadgk(integrand, t[i], t[i + 1],
                                                   atol = abstol, rtol = reltol)[1]
                end
                if t[i] == t[i + 1]
                    integrand = update_integrand_and_dgrad(res, sensealg, callback,
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

    _du .= adj_sol[end]
    _dp .= res'
    return true
end
function __adjoint_sensitivities!(_du::AbstractVector,
                                  _dp::AbstractVector,
                                  sol::ODESolution,
                                  sensealg::GaussAdjoint,
                                  t::Vector{Float64},
                                  solver::SciMLAlgorithm,
                                  abstol::Float64,
                                  reltol::Float64,
                                  callback::SciMLBase.DECallback,
                                  compute_∂G∂u::F;
                                  kwargs...)::Bool where {F}
    checkpoints = sol.t
    integrand = SciMLSensitivity.GaussIntegrand(sol, sensealg, checkpoints, nothing)
    integrand_values = DiffEqCallbacks.IntegrandValuesSum(SciMLSensitivity.allocate_zeros(sol.prob.p))
    cb = DiffEqCallbacks.IntegratingSumCallback((out, u, t, integrator) -> integrand(out, t,
                                                                                     u),
                                                integrand_values,
                                                SciMLSensitivity.allocate_vjp(sol.prob.p))
    rcb = nothing
    cb2 = nothing
    adj_prob = nothing

    adj_prob, cb2, rcb = ODEAdjointProblem(sol, sensealg, solver, integrand, t,
                                           compute_∂G∂u,
                                           nothing, nothing, nothing, nothing, Val(true);
                                           checkpoints = checkpoints,
                                           callback = callback, abstol = abstol,
                                           reltol = reltol)

    tstops = SciMLSensitivity.ischeckpointing(sensealg, sol) ? checkpoints :
             similar(sol.t, 0)

    adj_sol = solve(adj_prob, solver; abstol = abstol, reltol = reltol,
                    save_everystep = false,
                    save_start = false, save_end = true, saveat = eltype(sol.u[1])[],
                    tstops = tstops,
                    callback = CallbackSet(cb, cb2), kwargs...)
    res = integrand_values.integrand

    _du .= adj_sol[end]
    _dp .= SciMLSensitivity.__maybe_adjoint(res)'
    return true
end
