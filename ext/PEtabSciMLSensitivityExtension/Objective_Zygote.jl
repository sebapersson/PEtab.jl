function compute_cost_zygote(θ_est,
                             ode_problem::ODEProblem,
                             petab_model::PEtabModel,
                             simulation_info::PEtab.SimulationInfo,
                             θ_indices::PEtab.ParameterIndices,
                             measurement_info::PEtab.MeasurementsInfo,
                             parameter_info::PEtab.ParametersInfo,
                             solve_ode_condition::Function,
                             prior_info::PEtab.PriorInfo)
    θ_dynamic, θ_observable, θ_sd, θ_non_dynamic = PEtab.splitθ(θ_est, θ_indices)

    cost = _compute_cost_zygote(θ_dynamic, θ_sd, θ_observable, θ_non_dynamic, ode_problem,
                                petab_model, simulation_info, θ_indices, measurement_info,
                                parameter_info, solve_ode_condition)

    if prior_info.has_priors == true
        θ_estT = transformθ_zygote(θ_est, θ_indices.xnames, parameter_info)
        cost -= compute_priors(θ_est, θ_estT, θ_indices.xnames, prior_info)
    end

    return cost
end

# Computes the likelihood in such a in a Zygote compatible way, which mainly means that no arrays are mutated.
function _compute_cost_zygote(θ_dynamic,
                              θ_sd,
                              θ_observable,
                              θ_non_dynamic,
                              ode_problem::ODEProblem,
                              petab_model::PEtabModel,
                              simulation_info::PEtab.SimulationInfo,
                              θ_indices::PEtab.ParameterIndices,
                              measurement_info::PEtab.MeasurementsInfo,
                              parameter_info::PEtab.ParametersInfo,
                              solve_ode_condition::Function)::Real
    θ_dynamicT = transformθ_zygote(θ_dynamic, θ_indices.xids[:dynamic], parameter_info)
    θ_sdT = transformθ_zygote(θ_sd, θ_indices.xids[:noise], parameter_info)
    θ_observableT = transformθ_zygote(θ_observable, θ_indices.xids[:observable],
                                      parameter_info)
    θ_non_dynamicT = transformθ_zygote(θ_non_dynamic, θ_indices.xids[:nondynamic],
                                       parameter_info)

    _p, _u0 = PEtab.change_ode_parameters(ode_problem.p, θ_dynamicT, θ_indices, petab_model)
    _ode_problem = remake(ode_problem, p = _p, u0 = _u0)

    # Compute y_model and sd-val by looping through all experimental conditons. At the end
    # update the likelihood
    cost = 0.0
    for experimental_condition_id in simulation_info.conditionids[:experiment]
        tmax = simulation_info.tmaxs[experimental_condition_id]
        ode_sol, success = solve_ode_condition(_ode_problem, experimental_condition_id,
                                               θ_dynamicT, tmax)
        if success != true
            return Inf
        end

        cost += PEtab.cost_condition(ode_sol, _p, θ_sdT, θ_observableT,
                                             θ_non_dynamicT, petab_model,
                                             experimental_condition_id, θ_indices,
                                             measurement_info, parameter_info,
                                             simulation_info,
                                             compute_gradient_θ_dynamic_zygote = true)

        if isinf(cost)
            return cost
        end
    end

    return cost
end

# Solve the ODE system for one experimental conditions in a Zygote compatible manner. Not well maintained and lacks
# full support because Zygote code is currently the slowest (by far)
function solve_ode_condition_zygote(ode_problem::ODEProblem,
                                    experimental_id::Symbol,
                                    θ_dynamic,
                                    t_max,
                                    changeToExperimentalCondUsePre::Function,
                                    measurement_info::PEtab.MeasurementsInfo,
                                    simulation_info::PEtab.SimulationInfo,
                                    solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                    absTol::Float64,
                                    relTol::Float64,
                                    abstol_ss::Float64,
                                    reltol_ss,
                                    sensealg,
                                    compute_tstops::Function)

    # For storing ODE solution (required for split gradient computations)
    whichCondID = findfirst(x -> x == experimental_id,
                            simulation_info.conditionids[:experiment])

    # In case the model is first simulated to a steady state
    local success = true
    if simulation_info.has_pre_equilibration == true
        first_expid = simulation_info.conditionids[:pre_equilibration][whichCondID]
        shift_expid = simulation_info.conditionids[:simulation][whichCondID]
        t_save = simulation_info.tsaves[experimental_id]

        u0_pre = ode_problem.u0[:]
        pUsePre, u0UsePre = changeToExperimentalCondUsePre(ode_problem.p, ode_problem.u0,
                                                           first_expid, θ_dynamic)
        probUsePre = remake(ode_problem, tspan = (0.0, 1e8),
                            u0 = convert.(eltype(θ_dynamic), u0UsePre),
                            p = convert.(eltype(θ_dynamic), pUsePre))
        ssProb = SteadyStateProblem(probUsePre)
        solSS = solve(ssProb, DynamicSS(solver), abstol = abstol_ss, reltol = reltol_ss,
                      odesolve_kwargs = (; abstol = absTol, reltol = relTol))

        # Terminate if a steady state was not reached in preequilibration simulations
        if solSS.retcode != ReturnCode.Success
            return solSS, false
        end

        # Change to parameters for the post steady state parameters
        pUsePost, u0UsePostTmp = changeToExperimentalCondUsePre(ode_problem.p,
                                                                ode_problem.u0, shift_expid,
                                                                θ_dynamic)

        # Given the standard the experimentaCondition-file can change the initial values for a state
        # whose value was changed in the preequilibration-simulation. The experimentalCondition
        # value is prioritized by only changing u0 to the steady state value for those states
        # that were not affected by change to shift_expid.
        has_not_changed = (u0UsePostTmp .== u0_pre)
        u0UsePost = [has_not_changed[i] == true ? solSS[i] : u0UsePostTmp[i]
                     for i in eachindex(u0UsePostTmp)]
        probUsePost = remake(ode_problem, tspan = (0.0, t_max),
                             u0 = convert.(eltype(θ_dynamic), u0UsePost),
                             p = convert.(eltype(θ_dynamic), pUsePost))

        # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
        # The preequilibration simulations are terminated upon a steady state using the TerminateSteadyState callback.
        tstops = compute_tstops(probUsePost.u0, probUsePost.p)
        sol = solve(probUsePost,
                    solver,
                    abstol = absTol,
                    reltol = relTol,
                    saveat = t_save,
                    sensealg = sensealg,
                    callback = simulation_info.callbacks[experimental_id],
                    tstops = tstops)

        ChainRulesCore.@ignore_derivatives simulation_info.odesols[experimental_id] = sol

        if sol.retcode != ReturnCode.Success
            sucess = false
        end

        # In case the model is not first simulated to a steady state
    elseif simulation_info.has_pre_equilibration == false
        first_expid = simulation_info.conditionids[:simulation][whichCondID]
        t_save = simulation_info.tsaves[experimental_id]
        t_max_use = isinf(t_max) ? 1e8 : t_max

        pUse, u0Use = changeToExperimentalCondUsePre(ode_problem.p, ode_problem.u0,
                                                     first_expid, θ_dynamic)
        probUse = remake(ode_problem, tspan = (0.0, t_max_use))

        # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
        # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
        tstops = compute_tstops(probUse.u0, probUse.p)
        #tstops = Float64[]
        if !(typeof(solver) <: Vector{Symbol}) && isinf(t_max)
            sol = (probArg) -> solve(probArg, solver, abstol = absTol, reltol = relTol,
                                     save_on = false,
                                     save_end = true, dense = dense,
                                     callback = TerminateSteadyState(abstol_ss, reltol_ss))

        elseif !(typeof(solver) <: Vector{Symbol}) && !isinf(t_max)
            sol = solve(probUse, solver, p = pUse, u0 = u0Use,
                        abstol = absTol, reltol = relTol, saveat = t_save,
                        sensealg = sensealg,
                        callback = simulation_info.callbacks[experimental_id],
                        tstops = tstops)
        else
            println("Error : Solver option does not exist")
        end

        ChainRulesCore.@ignore_derivatives simulation_info.odesols[experimental_id] = sol

        if typeof(sol) <: ODESolution &&
           !(sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated)
            sucess = false
        end
    end

    return sol, success
end
