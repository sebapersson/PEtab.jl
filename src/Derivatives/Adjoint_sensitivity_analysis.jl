# Compute the adjoint gradient across all experimental conditions
function computeGradientAdjointDynamicθ(gradient::Vector{Float64},
                                        θ_dynamic::Vector{Float64},
                                        θ_sd::Vector{Float64},
                                        θ_observable::Vector{Float64},
                                        θ_nonDynamic::Vector{Float64},
                                        odeProblem::ODEProblem,
                                        odeSolverOptions::ODESolverOptions,
                                        ssSolverOptions::SteadyStateSolverOptions,
                                        sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                        petabModel::PEtabModel,
                                        simulationInfo::SimulationInfo,
                                        θ_indices::ParameterIndices,
                                        measurementInfo ::MeasurementsInfo,
                                        parameterInfo::ParametersInfo,
                                        petabODECache::PEtabODEProblemCache,
                                        petabODESolverCache::PEtabODESolverCache;
                                        sensealgSS=SteadyStateAdjoint(),
                                        expIDSolve::Vector{Symbol} = [:all])

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices, :θ_dynamic, petabODECache)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices, :θ_sd, petabODECache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices, :θ_observable, petabODECache)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices, :θ_nonDynamic, petabODECache)

    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamicT), odeProblem.p), u0 = convert.(eltype(θ_dynamicT), odeProblem.u0))
    changeODEProblemParameters!(_odeProblem.p, _odeProblem.u0, θ_dynamicT, θ_indices, petabModel)
    success = solveODEAllExperimentalConditions!(simulationInfo.odeSolutionsDerivatives, _odeProblem, petabModel, θ_dynamicT, petabODESolverCache, simulationInfo, θ_indices, odeSolverOptions, ssSolverOptions, expIDSolve=expIDSolve, denseSolution=true, onlySaveAtObservedTimes=false, trackCallback=true)
    if success != true
        gradient .= 1e8
        return
    end

    # In case of PreEq-critera we need to compute the pullback function at tSS to compute the VJP between
    # λ_t0 and the sensitivites at steady state time
    if simulationInfo.haspreEquilibrationConditionId == true
        evalVJPSSVec = generateVJPSSFunction(simulationInfo, sensealgSS, odeSolverOptions, ssSolverOptions, expIDSolve)
    end

    fill!(gradient, 0.0)
    # Compute the gradient by looping through all experimental conditions.
    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalConditionId = simulationInfo.experimentalConditionId[i]
        simulationConditionId = simulationInfo.simulationConditionId[i]

        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        if simulationInfo.haspreEquilibrationConditionId == true
            evalVJPSS = evalVJPSSVec[simulationInfo.preEquilibrationConditionId[i]]
        else
            evalVJPSS = identity
        end

        # In case the model is simulated first to a steady state we need to keep track of the post-equlibrium experimental
        # condition Id to identify parameters specific to an experimental condition.
        sol = simulationInfo.odeSolutionsDerivatives[experimentalConditionId]
        success = computeGradientAdjointExpCond!(gradient, sol, petabODECache, sensealg, odeSolverOptions,
                                                 θ_dynamicT, θ_sdT, θ_observableT, θ_nonDynamicT, experimentalConditionId,
                                                 simulationConditionId, simulationInfo,
                                                 petabModel, θ_indices, measurementInfo, parameterInfo, evalVJPSS)

        if success == false
            gradient .= 1e8
            return
        end
    end
    return
end


function generateVJPSSFunction(simulationInfo::SimulationInfo,
                               sensealgSS::SteadyStateAdjoint,
                               odeSolverOptions::ODESolverOptions,
                               ssSolverOptions::SteadyStateSolverOptions,
                               expIDSolve::Vector{Symbol})::NamedTuple

    # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
    # (expIDSolve != [["all]]) the number of preEq cond. might be smaller than the
    # total number of preEq cond.
    if expIDSolve[1] == :all
        preEquilibrationConditionId = unique(simulationInfo.preEquilibrationConditionId)
    else
        whichIds  = findall(x -> x ∈ simulationInfo.experimentalConditionId, expIDSolve)
        preEquilibrationConditionId = unique(simulationInfo.preEquilibrationConditionId[whichIds])
    end

    _evalVJPSS = Vector{Function}(undef, length(preEquilibrationConditionId))
    solver, abstol, reltol, force_dtmin, dtmin, maxiters = odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.force_dtmin, odeSolverOptions.dtmin, odeSolverOptions.maxiters
    for i in eachindex(preEquilibrationConditionId)

        odeProblem = simulationInfo.odePreEqulibriumSolutions[preEquilibrationConditionId[i]].prob
        ssOdeProblem = SteadyStateProblem(odeProblem)
        ySS, _evalVJPSSi = Zygote.pullback((p) ->    (
                                                      solve(ssOdeProblem,
                                                            DynamicSS(solver, abstol=ssSolverOptions.abstol, reltol=ssSolverOptions.reltol),
                                                            abstol=abstol,
                                                            reltol=reltol,
                                                            maxiters=maxiters,
                                                            force_dtmin=force_dtmin,
                                                            p=p,
                                                            sensealg=sensealgSS)[:]), odeProblem.p)

        _evalVJPSS[i] = (du) -> begin return _evalVJPSSi(du)[1] end
    end

    evalVJPSS = Tuple(f for f in _evalVJPSS)
    return NamedTuple{Tuple(name for name in preEquilibrationConditionId)}(evalVJPSS)
end
function generateVJPSSFunction(simulationInfo::SimulationInfo,
                               sensealgSS::Union{QuadratureAdjoint, InterpolatingAdjoint},
                               odeSolverOptions::ODESolverOptions,
                               ssSolverOptions::SteadyStateSolverOptions,
                               expIDSolve::Vector{Symbol})::NamedTuple

    # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
    # (expIDSolve != [["all]]) the number of preEq cond. might be smaller than the
    # total number of preEq cond.
    if expIDSolve[1] == :all
        preEquilibrationConditionId = unique(simulationInfo.preEquilibrationConditionId)
    else
        whichIds  = findall(x -> x ∈ simulationInfo.experimentalConditionId, expIDSolve)
        preEquilibrationConditionId = unique(simulationInfo.preEquilibrationConditionId[whichIds])
    end

    _evalVJPSS = Vector{Function}(undef, length(preEquilibrationConditionId))
    solver, abstol, reltol, force_dtmin, dtmin, maxiters = odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.force_dtmin, odeSolverOptions.dtmin, odeSolverOptions.maxiters
    for i in eachindex(preEquilibrationConditionId)

        # Sets up a function which takes du and solves the Adjoint ODE system with du
        # as starting point. This is a temporary ugly solution as there are some problems
        # with retcode Terminated and using CVODE_BDF
        _sol = simulationInfo.odePreEqulibriumSolutions[preEquilibrationConditionId[i]]
        _prob = remake(_sol.prob, tspan=(0.0, _sol.t[end]))
        sol = solve(_prob, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, sensealgSS, maxiters=maxiters)

        _evalVJPSSi = (du) -> computeVJPSS(du, sol, solver, sensealgSS, reltol, abstol, dtmin, force_dtmin, maxiters)
        _evalVJPSS[i] = _evalVJPSSi
    end

    evalVJPSS = Tuple(f for f in _evalVJPSS)
    return NamedTuple{Tuple(name for name in preEquilibrationConditionId)}(evalVJPSS)
end


# Compute the adjoint VJP for steady state simulated models via QuadratureAdjoint and InterpolatingAdjoint
# by, given du as initial values, solve the adjoint integral.
# TODO : Add interface for SteadyStateAdjoint
function computeVJPSS(du::AbstractVector,
                      _sol::ODESolution,
                      odeSolver::SciMLAlgorithm,
                      sensealg::QuadratureAdjoint,
                      reltol::Float64,
                      abstol::Float64,
                      dtmin::Union{Float64, Nothing},
                      force_dtmin::Bool,
                      maxiters::Int64)

    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, odeSolver, [_sol.t[end]], compute∂g∂uEmpty, nothing,
                                      nothing, nothing, nothing, Val(true))
    adj_prob.u0 .= du
    adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = true, save_start = true)
    integrand = AdjointSensitivityIntegrand(_sol, adj_sol, sensealg, nothing)
    res, err = SciMLSensitivity.quadgk(integrand, _sol.prob.tspan[1], _sol.t[end],
                                       atol = abstol, rtol = reltol)
    return res'
end
function computeVJPSS(du::AbstractVector,
                      _sol::ODESolution,
                      odeSolver::SciMLAlgorithm,
                      sensealg::InterpolatingAdjoint,
                      reltol::Float64,
                      abstol::Float64,
                      dtmin::Union{Float64, Nothing},
                      force_dtmin::Bool,
                      maxiters::Int64)

    nModelStates = length(_sol.prob.u0)
    adj_prob, rcb = ODEAdjointProblem(_sol, sensealg, odeSolver, [_sol.t[end]], compute∂g∂uEmpty, nothing,
                                      nothing, nothing, nothing, Val(true))

    adj_prob.u0[1:nModelStates] .= du[1:nModelStates]

    adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = true, save_start = true)
    out = adj_sol[end][(nModelStates+1):end]
    return out
end


function compute∂g∂uEmpty(out, u, p, t, i)
    out .= 0.0
end


# For a given experimental condition compute the gradient using adjoint sensitivity analysis
# for a funciton on the form G = (h - yObs)^2 / σ^2
# TODO : Important function - improve documentation.
function computeGradientAdjointExpCond!(gradient::Vector{Float64},
                                        sol::ODESolution,
                                        petabODECache::PEtabODEProblemCache,
                                        sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                        odeSolverOptions::ODESolverOptions,
                                        θ_dynamic::Vector{Float64},
                                        θ_sd::Vector{Float64},
                                        θ_observable::Vector{Float64},
                                        θ_nonDynamic::Vector{Float64},
                                        experimentalConditionId::Symbol,
                                        simulationConditionId::Symbol,
                                        simulationInfo::SimulationInfo,
                                        petabModel::PEtabModel,
                                        θ_indices::ParameterIndices,
                                        measurementInfo::MeasurementsInfo,
                                        parameterInfo::ParametersInfo,
                                        evalVJPSS::Function)::Bool

    # Extract experimetnalCondition specific parameter required to solve the
    # adjoitn ODE
    iPerTimePoint = simulationInfo.iPerTimePoint[experimentalConditionId]
    timeObserved = simulationInfo.timeObserved[experimentalConditionId]
    callback = simulationInfo.trackedCallbacks[experimentalConditionId]

    compute∂G∂u! = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint,
                                                         measurementInfo, parameterInfo,
                                                         θ_indices, petabModel,
                                                         θ_sd, θ_observable, θ_nonDynamic,
                                                         petabODECache.∂h∂u, petabODECache.∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p! = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint,
                                                         measurementInfo, parameterInfo,
                                                         θ_indices, petabModel,
                                                         θ_sd, θ_observable, θ_nonDynamic,
                                                         petabODECache.∂h∂p, petabODECache.∂σ∂p, compute∂G∂U=false)
                                        end

    solver, abstol, reltol, force_dtmin, dtmin, maxiters = odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.force_dtmin, odeSolverOptions.dtmin, odeSolverOptions.maxiters

    # The standard allow cases where we only observe data at t0, that is we do not solve the ODE. Here adjoint_sensitivities fails (naturally). In this case we compute the gradient
    # via ∇G_p = dp + du*J(u(t_0)) where du is the cost function differentiated with respect to the states at time zero,
    # dp is the cost function  with respect to the parameters at time zero and J is sensititvites at time
    # zero. Overall, the only workflow that changes below is that we compute du outside of the adjoint interface
    # and use sol[:] as we no longer can interpolate from the forward solution.
    onlyObsAtZero::Bool = false
    du = petabODECache.du
    dp = petabODECache.dp
    if !(length(timeObserved) == 1 && timeObserved[1] == 0.0)

        status = __adjoint_sensitivities!(du, dp, sol, sensealg, timeObserved, solver, abstol, reltol,
                                          force_dtmin, dtmin, maxiters, callback, compute∂G∂u!)
        status == false && return false
    else
        compute∂G∂u(du, sol[1], sol.prob.p, 0.0, 1)
        onlyObsAtZero = true
    end
    # Technically we can pass compute∂G∂p above to dgdp_discrete. However, odeProb.p often contain
    # constant parameters which are not a part ode the parameter estimation problem. Sometimes
    # the gradient for these evaluate to NaN (as they where never thought to be estimated) which
    # results in the entire gradient evaluating to NaN. Hence, we perform this calculation outside
    # of the lower level interface.
    ∂G∂p, ∂G∂p_ = petabODECache.∂G∂p_, petabODECache.∂G∂p
    fill!(∂G∂p, 0.0)
    for i in eachindex(timeObserved)
        if onlyObsAtZero == false
            compute∂G∂p!(∂G∂p_, sol(timeObserved[i]), sol.prob.p, timeObserved[i], i)
        else
            compute∂G∂p!(∂G∂p_, sol[1], sol.prob.p, timeObserved[i], i)
        end
        ∂G∂p .+= ∂G∂p_
    end

    _gradient = petabODECache._gradientAdjoint
    if simulationInfo.haspreEquilibrationConditionId == false
        # In case we do not simulate the ODE for a steady state first we can compute
        # the initial sensitivites easily via automatic differantitatiom
        St0 = petabODECache.St0
        ForwardDiff.jacobian!(St0, petabModel.compute_u0!, sol.prob.u0, sol.prob.p)
        _gradient .= dp .+ transpose(St0) * du

    else
        # In case we simulate to a stady state we need to compute a VJP. We use
        # Zygote pullback to avoid having to having build the Jacobian, rather
        # we create the yBar function required for the vector Jacobian product.
        @views _gradient .= dp .+ evalVJPSS(du)
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    adjustGradientTransformedParameters!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices,
                                         simulationConditionId, adjoint=true)
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
    adj_sol = solve(adjProb, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = false, save_start = false, saveat = eltype(sol[1])[],
                    tstops=tstops)
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
    adj_sol = solve(adj_prob, odeSolver; abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters,
                    save_everystep = true, save_start = true)

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
