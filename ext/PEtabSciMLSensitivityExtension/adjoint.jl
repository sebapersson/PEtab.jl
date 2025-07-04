# Compute gradient via adjoint sensitivity analysis
function grad_adjoint!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode!::Function,
                       probinfo::PEtab.PEtabODEProblemInfo, model_info::PEtab.ModelInfo;
                       cids::Vector{Symbol} = [:all])::Nothing where {T <: AbstractFloat}
    @unpack simulation_info, simulation_info, xindices, priors = model_info
    @unpack cache = probinfo
    PEtab.split_x!(x, xindices, cache)
    @unpack xdynamic_grad, xnotode_grad = cache

    _grad_adjoint_xdynamic!(xdynamic_grad, probinfo, model_info; cids = cids)
    @views grad[xindices.xindices[:dynamic]] .= xdynamic_grad

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if !isempty(xdynamic_grad) && all(xdynamic_grad .== 0.0)
        fill!(grad, 0.0)
        return nothing
    end

    # None-dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_notode = @view x[xindices.xindices[:not_system]]
    ReverseDiff.gradient!(xnotode_grad, _nllh_not_solveode!, x_notode)
    @views grad[xindices.xindices[:not_system]] .= xnotode_grad
    return nothing
end

function _grad_adjoint_xdynamic!(grad::Vector{<:AbstractFloat},
                                 probinfo::PEtab.PEtabODEProblemInfo,
                                 model_info::PEtab.ModelInfo;
                                 cids::Vector{Symbol} = [:all])::Nothing
    @unpack cache, sensealg, sensealg_ss = probinfo
    @unpack simulation_info, xindices = model_info
    xnoise_ps = PEtab.transform_x(cache.xnoise, xindices, :xnoise, cache)
    xobservable_ps = PEtab.transform_x(cache.xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(cache.xnondynamic, xindices, :xnondynamic, cache)
    xdynamic_ps = PEtab.transform_x(cache.xdynamic, xindices, :xdynamic, cache)

    success = PEtab.solve_conditions!(model_info, xdynamic_ps, probinfo; cids = cids,
                                      dense_sol = true, save_observed_t = false,
                                      track_callback = true)
    if success == false
        fill!(grad, 0.0)
        return nothing
    end

    # In case of pre-equilibration a VJP between λt0 and the sensitivites at steady state
    # must be computed.
    if simulation_info.has_pre_equilibration == true
        vjps_ss = _get_vjps_ss(probinfo, simulation_info, sensealg_ss, cids)
    end

    fill!(grad, 0.0)
    for icid in eachindex(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(imulation_info.conditionids[:experiment][cid] in cids)
            continue
        end

        if simulation_info.has_pre_equilibration == true
            vjp_cid_ss = vjps_ss[simulation_info.conditionids[:pre_equilibration][icid]]
        else
            vjp_cid_ss = identity
        end

        success = _grad_adjoint_cond!(grad, xdynamic_ps, xnoise_ps, xobservable_ps,
                                      xnondynamic_ps, icid, probinfo, model_info,
                                      vjp_cid_ss)
        if success == false
            fill!(grad, 0.0)
            return nothing
        end
    end
    return nothing
end

# TODO: Figure out how to get the math behind SteadyStateAdjoint working, should really
# be easy as it boils down to a single Matrix operation given Jacobian
function _get_vjps_ss(probinfo::PEtab.PEtabODEProblemInfo,
                      simulation_info::PEtab.SimulationInfo,
                      sensealg_ss::AdjointAlg, cids::Vector{Symbol})::Dict{Symbol, Function}
    if cids[1] == :all
        preeq_ids = unique(simulation_info.conditionids[:pre_equilibration])
    else
        which_ids = findall(x -> x in simulation_info.conditionids[:experiment], cids)
        preeq_ids = unique(simulation_info.conditionids[:pre_equilibration][which_ids])
    end

    # The VJP function takes du as input and solves the Adjoint ODESystem with du as
    # starting point. The only drawback with this approach computationally is that
    # recomputation is needed for each du.
    vjps_ss = Dict{Symbol, Function}()
    @unpack solver, abstol, reltol, force_dtmin, maxiters = probinfo.solver_gradient
    for preeq_id in preeq_ids
        # The current solution below with resolving forward is not needed, but as
        # CVODE does not work with retcode = Terminated it is currently used.
        # TODO: Fix not resolve (deepcopy should do), and to set retcode manually in
        # the SS-callback
        _sol = simulation_info.odesols_preeq[preeq_id]
        _prob = remake(_sol.prob, tspan = (0.0, _sol.t[end]))
        sol = solve(_prob, solver, abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters)
        vjp_cid = (du) -> VJP_ss(du, sol, solver, sensealg_ss, reltol, abstol, force_dtmin,
                                 maxiters)
        vjps_ss[preeq_id] = vjp_cid
    end
    return vjps_ss
end

# TODO : Add interface for SteadyStateAdjoint
function VJP_ss(du::AbstractVector, _sol::ODESolution, solver::SciMLAlgorithm,
                sensealg::QuadratureAdjoint, reltol::Float64, abstol::Float64,
                force_dtmin::Bool, maxiters::Int64)::AbstractVector
    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, solver, [_sol.t[end]],
                                      ∂g∂u_empty!, nothing, nothing, nothing,
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
function VJP_ss(du::AbstractVector, _sol::ODESolution, solver::SciMLAlgorithm,
                sensealg::InterpolatingAdjoint, reltol::Float64, abstol::Float64,
                force_dtmin::Bool, maxiters::Int64)::AbstractVector
    n_model_states = length(_sol.prob.u0)
    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, solver, [_sol.t[end]],
                                      ∂g∂u_empty!, nothing, nothing, nothing,
                                      nothing, Val(true))

    adj_prob.u0[1:n_model_states] .= du[1:n_model_states]

    adj_sol = solve(adj_prob, solver; abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters,
                    save_everystep = true, save_start = true)
    out = adj_sol[end][(n_model_states + 1):end]
    return out
end

function _grad_adjoint_cond!(grad::Vector{T}, xdynamic::Vector{T}, xnoise::Vector{T},
                             xobservable::Vector{T}, xnondynamic::Vector{T}, icid::Int64,
                             probinfo::PEtab.PEtabODEProblemInfo,
                             model_info::PEtab.ModelInfo,
                             vjp_ss_cid::Function)::Bool where {T <: AbstractFloat}
    @unpack xindices, simulation_info, model = model_info
    @unpack petab_parameters, petab_measurements = model_info
    @unpack imeasurements_t, tsaves, smatrixindices, tracked_callbacks = simulation_info
    @unpack sensealg, cache, solver_gradient = probinfo
    @unpack solver_adj, abstol_adj, reltol_adj, maxiters, force_dtmin = solver_gradient

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    sol = simulation_info.odesols_derivatives[cid]
    callback = tracked_callbacks[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = PEtab._get_∂G∂_!(probinfo, model_info, cid, xnoise, xobservable,
                                    xnondynamic)

    # The PEtab standard allow cases where we only observe data at t0, that is we do not
    # solve the ODE. Here adjoint_sensitivities fails (naturally). In this case we compute
    # the gradient via ∇G_p = dp + du*J(u(t_0)) where du is the cost function differentiated
    # with respect to the states at time zero, dp is the cost function  with respect to the
    # parameters at time zero and J is sensititvites at time zero. Overall, the only
    # workflow that changes below is that we compute du outside of the adjoint interface
    # and use sol[:] as we no longer can interpolate from the forward solution.
    only_obs_at_zero::Bool = false
    @unpack du, dp, ∂G∂p, ∂G∂p_, adjoint_grad, St0 = cache
    if length(tsaves[cid]) == 1 && tsaves[cid][1] == 0.0
        ∂G∂u!(du, sol[1], sol.prob.p, 0.0, 1)
        only_obs_at_zero = true
    else
        status = __adjoint_sensitivities!(du, dp, sol, sensealg, tsaves[cid], solver_adj,
                                          abstol_adj, reltol_adj, callback, ∂G∂u!;
                                          maxiters = maxiters, force_dtmin = force_dtmin)
        status == false && return status
    end
    # Technically we can pass compute∂G∂p above to dgdp_discrete. However, odeproblem.p
    # often contain constant parameters which are not a part ode the parameter estimation
    # problem. Sometimes the gradient for these evaluate to NaN (as they where never thought
    # to be estimated) which results in the entire gradient evaluating to NaN. Hence, we
    # perform this calculation outside of the lower level interface.
    fill!(∂G∂p, 0.0)
    for (it, tsave) in pairs(tsaves[cid])
        if only_obs_at_zero == false
            ∂G∂p!(∂G∂p_, sol(tsave), sol.prob.p, tsave, it)
        else
            ∂G∂p!(∂G∂p_, sol[1], sol.prob.p, tsave, it)
        end
        ∂G∂p .+= ∂G∂p_
    end
    # In case we do not simulate the ODE for a steady state first we can compute
    # the initial sensitivites easily via automatic differantitatiom
    if simulation_info.has_pre_equilibration == false
        ForwardDiff.jacobian!(St0, model.u0!, sol.prob.u0, sol.prob.p)
        adjoint_grad .= dp .+ transpose(St0) * du
        # In case we simulate to a stady state we need to compute a VJP.
    else
        adjoint_grad .= dp .+ vjp_ss_cid(du)
    end

    # Adjust if gradient is non-linear scale (e.g. log and log10).
    PEtab.grad_to_xscale!(grad, adjoint_grad, ∂G∂p, xdynamic, xindices, simid,
                          adjoint = true)
    return true
end

# In order to obtain the ret-codes when solving the adjoint ODE system we must, as here
# copy to 99% from SciMLSensitivity GitHub repo to access the actual solve-call to the
# ODEAdjointProblem. Under MIT Expat license at https://github.com/SciML/SciMLSensitivity.jl
function __adjoint_sensitivities!(_du::AbstractVector, _dp::AbstractVector,
                                  sol::ODESolution,
                                  sensealg::InterpolatingAdjoint, t::Vector{Float64},
                                  solver::SciMLAlgorithm, abstol::Float64, reltol::Float64,
                                  callback::SciMLBase.DECallback, compute_∂G∂u::F;
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

    adj_prob, cb2, rcb = ODEAdjointProblem(sol, sensealg, solver, integrand, cb, t,
                                           compute_∂G∂u, nothing, nothing, nothing, nothing,
                                           Val(true); checkpoints = checkpoints,
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

function ∂g∂u_empty!(out, u, p, t, i)::Nothing
    fill!(out, 0.0)
    return nothing
end

function PEtab._get_cbs(odeproblem::ODEProblem, simulation_info::PEtab.SimulationInfo,
                        simid::Symbol, sensealg::AdjointAlg)::SciMLBase.DECallback
    cbset = SciMLSensitivity.track_callbacks(simulation_info.callbacks[simid],
                                             odeproblem.tspan[1], odeproblem.u0,
                                             odeproblem.p, sensealg)
    simulation_info.tracked_callbacks[simid] = cbset
    return cbset
end
