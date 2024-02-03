#=
    Functionallity for solving a PEtab ODE-model across all experimental conditions. Code is compatible with ForwardDiff,
    and there is functionallity for computing the sensitivity matrix.
=#

function solve_ode_all_conditions!(ode_sols::Dict{Symbol, Union{Nothing, ODESolution}},
                                   ode_problem::ODEProblem,
                                   petab_model::PEtabModel,
                                   θ_dynamic::AbstractVector,
                                   petab_ODESolver_cache::PEtabODESolverCache,
                                   simulation_info::SimulationInfo,
                                   θ_indices::ParameterIndices,
                                   ode_solver::ODESolver,
                                   ss_solver::SteadyStateSolver;
                                   exp_id_solve::Vector{Symbol} = [:all],
                                   n_timepoints_save::Int64 = 0,
                                   save_at_observed_t::Bool = false,
                                   dense_sol::Bool = true,
                                   track_callback::Bool = false,
                                   compute_forward_sensitivites::Bool = false)::Bool # Required for adjoint sensitivity analysis
    change_simulation_condition! = (p_ode_problem, u0, conditionId) -> _change_simulation_condition!(p_ode_problem,
                                                                                                     u0,
                                                                                                     conditionId,
                                                                                                     θ_dynamic,
                                                                                                     petab_model,
                                                                                                     θ_indices,
                                                                                                     compute_forward_sensitivites = compute_forward_sensitivites)
    # In case the model is first simulated to a steady state
    if simulation_info.has_pre_equilibration_condition_id == true

        # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
        # (exp_id_solve != [["all]]) the number of preEq cond. might be smaller than the
        # total number of preEq cond.
        if exp_id_solve[1] == :all
            pre_equilibration_id = unique(simulation_info.pre_equilibration_condition_id)
        else
            which_id = findall(x -> x ∈ simulation_info.experimental_condition_id,
                               exp_id_solve)
            pre_equilibration_id = unique(simulation_info.pre_equilibration_condition_id[which_id])
        end

        # Arrays to store steady state (pre-eq) values.
        u_ss = Matrix{eltype(θ_dynamic)}(undef,
                                         (length(ode_problem.u0),
                                          length(pre_equilibration_id)))
        u_t0 = Matrix{eltype(θ_dynamic)}(undef,
                                         (length(ode_problem.u0),
                                          length(pre_equilibration_id)))

        for i in eachindex(pre_equilibration_id)
            _ode_problem = set_ode_parameters(ode_problem, petab_ODESolver_cache,
                                              pre_equilibration_id[i])
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                _ode_sols = simulation_info.ode_sols_pre_equlibrium
                _ode_sols[pre_equilibration_id[i]] = solve_ode_pre_equlibrium!((@view u_ss[:,
                                                                                           i]),
                                                                               (@view u_t0[:,
                                                                                           i]),
                                                                               _ode_problem,
                                                                               change_simulation_condition!,
                                                                               pre_equilibration_id[i],
                                                                               ode_solver,
                                                                               ss_solver,
                                                                               petab_model.convert_tspan)
            catch e
                check_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            if simulation_info.ode_sols_pre_equlibrium[pre_equilibration_id[i]].retcode !=
               ReturnCode.Terminated
                simulation_info.could_solve[1] = false
                return false
            end
        end
    end

    @inbounds for i in eachindex(simulation_info.experimental_condition_id)
        experimental_id = simulation_info.experimental_condition_id[i]

        if exp_id_solve[1] != :all && experimental_id ∉ exp_id_solve
            continue
        end

        # In case save_at_observed_t=true all other options are overridden and we only save the data
        # at observed time points.
        _tmax = simulation_info.tmax[experimental_id]
        _t_save = get_t_saveat(Val(save_at_observed_t), simulation_info, experimental_id,
                               _tmax, n_timepoints_save)
        _dense_sol = should_save_dense_sol(Val(save_at_observed_t), n_timepoints_save,
                                           dense_sol)
        _ode_problem = set_ode_parameters(ode_problem, petab_ODESolver_cache,
                                          experimental_id)

        # In case we have a simulation with PreEqulibrium
        if simulation_info.pre_equilibration_condition_id[i] != :None
            which_index = findfirst(x -> x ==
                                         simulation_info.pre_equilibration_condition_id[i],
                                    pre_equilibration_id)
            # See comment above on domain error
            try
                ode_sols[experimental_id] = solve_ode_post_equlibrium(_ode_problem,
                                                                      (@view u_ss[:,
                                                                                  which_index]),
                                                                      (@view u_t0[:,
                                                                                  which_index]),
                                                                      change_simulation_condition!,
                                                                      simulation_info,
                                                                      simulation_info.simulation_condition_id[i],
                                                                      experimental_id,
                                                                      _tmax,
                                                                      ode_solver,
                                                                      petab_model.compute_tstops,
                                                                      _t_save,
                                                                      _dense_sol,
                                                                      track_callback,
                                                                      petab_model.convert_tspan)
            catch e
                check_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            if ode_sols[experimental_id].retcode != ReturnCode.Success
                simulation_info.could_solve[1] = false
                return false
            end

            # In case we have an ODE solution without Pre-equlibrium
        else
            try
                ode_sols[experimental_id] = solve_ode_no_pre_equlibrium(_ode_problem,
                                                                        change_simulation_condition!,
                                                                        simulation_info,
                                                                        simulation_info.simulation_condition_id[i],
                                                                        ode_solver,
                                                                        _tmax,
                                                                        petab_model.compute_tstops,
                                                                        _t_save,
                                                                        _dense_sol,
                                                                        track_callback,
                                                                        petab_model.convert_tspan)
            catch e
                check_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            retcode = ode_sols[experimental_id].retcode
            if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
                simulation_info.could_solve[1] = false
                return false
            end
        end
    end

    return true
end
function solve_ode_all_conditions!(sol_values::AbstractMatrix,
                                   _θ_dynamic::AbstractVector,
                                   petab_ODESolver_cache::PEtabODESolverCache,
                                   ode_sols::Dict{Symbol, Union{Nothing, ODESolution}},
                                   ode_problem::ODEProblem,
                                   petab_model::PEtabModel,
                                   simulation_info::SimulationInfo,
                                   ode_solver::ODESolver,
                                   ss_solver::SteadyStateSolver,
                                   θ_indices::ParameterIndices,
                                   petab_ODE_cache::PEtabODEProblemCache;
                                   exp_id_solve::Vector{Symbol} = [:all],
                                   n_timepoints_save::Int64 = 0,
                                   save_at_observed_t::Bool = false,
                                   dense_sol::Bool = true,
                                   track_callback::Bool = false,
                                   compute_forward_sensitivites::Bool = false,
                                   compute_forward_sensitivites_ad::Bool = false)::Nothing
    if compute_forward_sensitivites_ad == true &&
       petab_ODE_cache.nθ_dynamic[1] != length(_θ_dynamic)
        θ_dynamic = _θ_dynamic[petab_ODE_cache.θ_dynamic_output_order]
    else
        θ_dynamic = _θ_dynamic
    end

    _ode_problem = remake(ode_problem, p = convert.(eltype(θ_dynamic), ode_problem.p),
                          u0 = convert.(eltype(θ_dynamic), ode_problem.u0))
    change_ode_parameters!(_ode_problem.p, _ode_problem.u0, θ_dynamic, θ_indices,
                           petab_model)

    sucess = solve_ode_all_conditions!(ode_sols,
                                       _ode_problem,
                                       petab_model,
                                       θ_dynamic,
                                       petab_ODESolver_cache,
                                       simulation_info,
                                       θ_indices,
                                       ode_solver,
                                       ss_solver,
                                       exp_id_solve = exp_id_solve,
                                       n_timepoints_save = n_timepoints_save,
                                       save_at_observed_t = save_at_observed_t,
                                       dense_sol = dense_sol,
                                       track_callback = track_callback,
                                       compute_forward_sensitivites = compute_forward_sensitivites)

    # Effectively we return a big-array with the ODE-solutions accross all experimental conditions, where
    # each column is a time-point.
    if sucess != true
        sol_values .= 0.0
        return nothing
    end

    # i_start and i_end tracks which entries in sol_values we store a specific experimental condition
    i_start, i_end = 1, 0
    for i in eachindex(simulation_info.experimental_condition_id)
        experimental_id = simulation_info.experimental_condition_id[i]
        i_end += length(simulation_info.time_observed[experimental_id])
        if exp_id_solve[1] == :all ||
           simulation_info.experimental_condition_id[i] ∈ exp_id_solve
            @views sol_values[:, i_start:i_end] .= Array(ode_sols[experimental_id])
        end
        i_start = i_end + 1
    end
    return nothing
end

function solve_ODE_all_conditions(ode_problem::ODEProblem,
                                  petab_model::PEtabModel,
                                  θ_dynamic::AbstractVector,
                                  petab_ODESolver_cache::PEtabODESolverCache,
                                  simulation_info::SimulationInfo,
                                  θ_indices::ParameterIndices,
                                  ode_solver::ODESolver,
                                  ss_solver::SteadyStateSolver;
                                  exp_id_solve::Vector{Symbol} = [:all],
                                  n_timepoints_save::Int64 = 0,
                                  save_at_observed_t::Bool = false,
                                  dense_sol::Bool = true,
                                  track_callback::Bool = false,
                                  compute_forward_sensitivites::Bool = false)::Tuple{Dict{Symbol,
                                                                                          Union{Nothing,
                                                                                                ODESolution}},
                                                                                     Bool}
    ode_sols = deepcopy(simulation_info.ode_sols)
    success = solve_ode_all_conditions!(ode_sols,
                                        ode_problem,
                                        petab_model,
                                        θ_dynamic,
                                        petab_ODESolver_cache,
                                        simulation_info,
                                        θ_indices,
                                        ode_solver,
                                        ss_solver,
                                        exp_id_solve = exp_id_solve,
                                        n_timepoints_save = n_timepoints_save,
                                        save_at_observed_t = save_at_observed_t,
                                        dense_sol = dense_sol,
                                        track_callback = track_callback,
                                        compute_forward_sensitivites = compute_forward_sensitivites)

    return ode_sols, success
end

function solve_ode_post_equlibrium(ode_problem::ODEProblem,
                                   u_ss::AbstractVector,
                                   u_t0::AbstractVector,
                                   change_simulation_condition!::Function,
                                   simulation_info::SimulationInfo,
                                   simulation_condition_id::Symbol,
                                   experimental_id::Symbol,
                                   tmax::Float64,
                                   ode_solver::ODESolver,
                                   compute_tstops::Function,
                                   t_save::Vector{Float64},
                                   dense_sol,
                                   track_callback,
                                   convert_tspan)::ODESolution
    change_simulation_condition!(ode_problem.p, ode_problem.u0, simulation_condition_id)
    # Sometimes the experimentaCondition-file changes the initial values for a state
    # whose value was changed in the preequilibration-simulation. The experimentaCondition
    # value is prioritized by only changing u0 to the steady state value for those states
    # that were not affected by change to shift_expid.
    has_not_changed = (ode_problem.u0 .== u_t0)
    @views ode_problem.u0[has_not_changed] .= u_ss[has_not_changed]

    # According to the PEtab standard we can sometimes have that initial assignment is overridden for
    # pre-eq simulation, but we do not want to override for main simulation which is done automatically
    # by change_simulation_condition!. These cases are marked as NaN
    is_nan = isnan.(ode_problem.u0)
    @views ode_problem.u0[is_nan] .= u_ss[is_nan]

    # Here it is IMPORTANT that we copy ode_problem.p[:] else different experimental conditions will
    # share the same parameter vector p. This will, for example, cause the lower level adjoint
    # sensitivity interface to fail.
    _ode_problem = get_tspan(ode_problem, tmax, ode_solver.solver, convert_tspan)
    @views _ode_problem.u0 .= ode_problem.u0[:] # This is needed due as remake does not work correctly for forward sensitivity equations

    # If case of adjoint sensitivity analysis we need to track the callback to get correct gradients
    tstops = compute_tstops(_ode_problem.u0, _ode_problem.p)
    callback_set = get_callbackset(_ode_problem, simulation_info, experimental_id,
                                   simulation_info.sensealg)
    sol = compute_ode_sol(_ode_problem, ode_solver.solver, ode_solver, ode_solver.abstol,
                          ode_solver.reltol,
                          t_save, dense_sol, callback_set, tstops)

    return sol
end

function solve_ode_no_pre_equlibrium(ode_problem::ODEProblem,
                                     change_simulation_condition!::F1,
                                     simulation_info::SimulationInfo,
                                     simulation_condition_id::Symbol,
                                     ode_solver::ODESolver,
                                     _tmax::Float64,
                                     compute_tstops::F2,
                                     t_save::Vector{Float64},
                                     dense_sol::Bool,
                                     track_callback::Bool,
                                     convert_tspan::Bool)::ODESolution where {
                                                                              F1 <:
                                                                              Function,
                                                                              F2 <:
                                                                              Function}

    # Change experimental condition
    tmax = isinf(_tmax) ? 1e8 : _tmax
    change_simulation_condition!(ode_problem.p, ode_problem.u0, simulation_condition_id)
    _ode_problem = get_tspan(ode_problem, tmax, ode_solver.solver, convert_tspan)
    @views _ode_problem.u0 .= ode_problem.u0[:] # Required remake does not handle Senstivity-problems correctly

    tstops = compute_tstops(_ode_problem.u0, _ode_problem.p)
    callback_set = get_callbackset(_ode_problem, simulation_info, simulation_condition_id,
                                   simulation_info.sensealg)
    sol = compute_ode_sol(_ode_problem, ode_solver.solver, ode_solver, ode_solver.abstol,
                          ode_solver.reltol,
                          t_save, dense_sol, callback_set, tstops)

    return sol
end

function compute_ode_sol(ode_problem::ODEProblem,
                         solver::S,
                         ode_solver::ODESolver,
                         abstol_ss::Float64,
                         reltol_ss::Float64,
                         t_save::Vector{Float64},
                         dense_sol::Bool,
                         callback_set::SciMLBase.DECallback,
                         tstops::AbstractVector)::ODESolution where {S <: SciMLAlgorithm}
    @unpack abstol, reltol, force_dtmin, dtmin, maxiters, verbose = ode_solver

    # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
    # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
    if isinf(ode_problem.tspan[2]) || ode_problem.tspan[2] == 1e8
        sol = solve(ode_problem, solver, abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters, save_on = false,
                    save_start = false, save_end = true, dense = dense_sol,
                    callback = TerminateSteadyState(abstol_ss, reltol_ss),
                    verbose = verbose)
    else
        sol = solve(ode_problem, solver, abstol = abstol, reltol = reltol,
                    force_dtmin = force_dtmin, maxiters = maxiters, saveat = t_save,
                    dense = dense_sol, tstops = tstops, callback = callback_set,
                    verbose = verbose)
    end
    return sol
end

function check_error(e)
    if e isa BoundsError
        @warn "Bounds error ODE solve"
    elseif e isa DomainError
        @warn "Domain error on ODE solve"
    elseif e isa SingularException
        @warn "Singular exception on ODE solve"
    else
        rethrow(e)
    end
end
