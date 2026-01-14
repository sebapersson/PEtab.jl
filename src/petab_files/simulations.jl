function SimulationInfo(cbs::Dict{Symbol, SciMLBase.DECallback}, petab_measurements::PEtabMeasurements, petab_events::Vector{PEtabEvent}; sensealg)::SimulationInfo
    conditionids = _get_conditionids(petab_measurements)
    has_pre_equilibration = !all(conditionids[:pre_equilibration] .== :None)

    # Precompute values needed by the ODE-solver, such as tmax, tsave, tstarts...
    # tsaves_no_cbs is needed to pre-allocate arrays of right save for ForwardEquation
    # gradients (as it needs to know the length of the ODESolution)
    tmaxs = _get_tmaxs(conditionids, petab_measurements)
    tstarts = _get_tstarts(conditionids, petab_measurements)
    tsaves = _get_tsaves(conditionids, petab_measurements, cbs, petab_events)
    tsaves_no_cbs = _get_tsaves(conditionids, petab_measurements, cbs, petab_events; exclude_events = true)

    # Indices for getting measurement points for each condition. The second is a vector
    # of vector that accounts for multiple measurements per time-point which needs to
    # be accounted for in gradient compuations
    imeasurements = _get_imeasurements(conditionids, petab_measurements)
    imeasurements_t = _get_imeasurements_t(imeasurements, petab_measurements)

    # Time indicies in ODESolution for each measurement
    imeasurements_t_sol = _get_imeasurements_t_sol(imeasurements, petab_measurements)

    # When computing forward sensitivities via forward mode automatic differentiation we get
    # a big sensitivity matrix across all experimental conditions, where S[i:(i+nStates)]
    # corresponds to the sensitivities at a specific time-point. Here the indices for each
    # experimental condition is computed. Note, this assumes that we solve the ODE:s in the
    # order of experimental_id (which is true)
    smatrixindices = _get_smatrixindices(conditionids[:experiment], tsaves_no_cbs)

    # When computing the gradient/Hessian for parameter not in the ODE-system with autodiff,
    # we do not need to solve (trace) the ODE solution. We only need an ODESolution.
    # Therefore the ODESolutions need to be stored in a Dict
    odesols = Dict{Symbol, ODESolution}()
    odesols_preeq = Dict{Symbol, Union{ODESolution, SciMLBase.NonlinearSolution}}()
    odesols_derivative = Dict{Symbol, ODESolution}()

    # Some models, e.g those with time dependent piecewise statements, have cbs encoded.
    # For adjoint sensitivity analysis these most be tracked
    tracked_cbs = Dict{Symbol, SciMLBase.DECallback}()
    conditions_df_ids = Iterators.flatten((conditionids[:simulation],
        filter(x -> x != :None, conditionids[:pre_equilibration])))
    for id in unique(conditions_df_ids)
        tracked_cbs[id] = deepcopy(cbs[id])
    end

    could_solve = [true]
    return SimulationInfo(conditionids, has_pre_equilibration, tstarts, tmaxs, tsaves,
                          tsaves_no_cbs, imeasurements, imeasurements_t, imeasurements_t_sol,
                          smatrixindices, odesols, odesols_derivative, odesols_preeq,
                          could_solve, cbs, tracked_cbs, sensealg)
end

function _get_conditionids(petab_measurements::PEtabMeasurements)::Dict{Symbol, Vector{Symbol}}
    @unpack pre_equilibration_condition_id, simulation_condition_id = petab_measurements
    # An experimental id is uniquely defined by a pre-equilibrium and simulation id, where
    # the former can be empty. For each experimental id we need to store the corresponding
    # pre and simulation id, and their concatenation
    pre_equilibration_ids = Symbol[]
    simulation_ids = Symbol[]
    experiment_ids = Symbol[]
    for i in eachindex(pre_equilibration_condition_id)
        measurement_preeq_id = pre_equilibration_condition_id[i]
        measurement_sim_id = simulation_condition_id[i]
        # In case of no pre-equilibration
        if measurement_preeq_id == :None
            measurement_sim_id in experiment_ids && continue
            push!(pre_equilibration_ids, :None)
            push!(simulation_ids, measurement_sim_id)
            push!(experiment_ids, measurement_sim_id)
            continue
        end

        # In case of pre-equilibration
        measurement_exp_id = prod(string.([measurement_preeq_id, measurement_sim_id])) |>
                             Symbol
        measurement_exp_id in experiment_ids && continue
        push!(pre_equilibration_ids, measurement_preeq_id)
        push!(simulation_ids, measurement_sim_id)
        push!(experiment_ids, measurement_exp_id)
    end
    return Dict(:pre_equilibration => pre_equilibration_ids, :simulation => simulation_ids,
                :experiment => experiment_ids)
end

function _get_tsaves(conditionids::Dict{Symbol, Vector{Symbol}}, petab_measurements::PEtabMeasurements, cbs::Dict{Symbol, SciMLBase.DECallback}, petab_events::Vector{PEtabEvent}; exclude_events::Bool = false)::Dict{Symbol, Vector{Float64}}
    tsave = Dict{Symbol, Vector{Float64}}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        pre_equilibration_id = conditionids[:pre_equilibration][i]
        simulation_id = conditionids[:simulation][i]
        it = _get_tindices(pre_equilibration_id, simulation_id, petab_measurements)
        tsave_experiment = sort(unique(petab_measurements.time[it]))

        # If a PEtabEvent is triggered at a tsave value, following the PEtab standard model
        # output should be saved after event completions. To this end, the trigger time
        # must be removed from tsave_experiment. Now, save_positions cannot be modified
        # in the callback as the callbacks parsed from SBML have initialize field, which
        # means that setting save_positions[2] = true results in an extra time-point
        # being saved a t0. Rather, the affect! function for PEtab events include a
        # save_u variable which can be set.
        if exclude_events == false
            i_events = _get_petab_events_simulation_id(petab_events, simulation_id)
            for j in eachindex(i_events)
                trigger_time = petab_events[i_events[j]].trigger_time
                !(trigger_time in tsave_experiment) && continue
                tsave_experiment = filter(x -> x != trigger_time, tsave_experiment)
                cbs[simulation_id].discrete_callbacks[j].affect!._save_u[1] = true
            end
        end

        tsave[experiment_id] = tsave_experiment
    end
    return tsave
end

function _get_tmaxs(conditionids::Dict{Symbol, Vector{Symbol}},
                    petab_measurements::PEtabMeasurements)::Dict{Symbol, Float64}
    tmaxs = Dict{Symbol, Float64}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        preeqids, cids = conditionids[:pre_equilibration][i], conditionids[:simulation][i]
        it = _get_tindices(preeqids, cids, petab_measurements)
        tmaxs[experiment_id] = maximum(petab_measurements.time[it])
    end
    return tmaxs
end

function _get_tstarts(conditionids::Dict{Symbol, Vector{Symbol}}, petab_measurements::PEtabMeasurements)::Dict{Symbol, Float64}
    tstarts = Dict{Symbol, Float64}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        preeqids, cids = conditionids[:pre_equilibration][i], conditionids[:simulation][i]
        it = _get_tindices(preeqids, cids, petab_measurements)
        tstarts[experiment_id] = petab_measurements.simulation_start_time[it[1]]
    end
    return tstarts
end

function _get_imeasurements(conditionids::Dict{Symbol, Vector{Symbol}},
                            petab_measurements::PEtabMeasurements)::Dict{Symbol,
                                                                         Vector{Int64}}
    imeasurements = Dict{Symbol, Vector{Int64}}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        preeqids, cids = conditionids[:pre_equilibration][i], conditionids[:simulation][i]
        it = _get_tindices(preeqids, cids, petab_measurements)
        imeasurements[experiment_id] = it
    end
    return imeasurements
end

function _get_tindices(pre_equilibration_id::Symbol, simulation_id::Symbol,
                       petab_measurements::PEtabMeasurements)::Vector{Int64}
    @unpack pre_equilibration_condition_id, simulation_condition_id = petab_measurements
    ipreeq = findall(x -> x == pre_equilibration_id, pre_equilibration_condition_id)
    isim = findall(x -> x == simulation_id, simulation_condition_id)
    return intersect(ipreeq, isim)
end

function _get_imeasurements_t_sol(imeasurements::Dict{Symbol, Vector{Int64}},
                                  petab_measurements::PEtabMeasurements)::Vector{Int64}
    time = petab_measurements.time
    imeasurements_t_sol = zeros(Int64, length(time))
    for i in eachindex(time)
        for ims in values(imeasurements)
            timepoints_sorted = unique(sort(time[ims]))
            for im in ims
                im != i && continue
                imeasurements_t_sol[i] = findfirst(x -> x == time[i], timepoints_sorted)
                break
            end
        end
    end
    return imeasurements_t_sol
end

function _get_imeasurements_t(imeasurements::Dict{Symbol, Vector{Int64}},
                              petab_measurements::PEtabMeasurements)::Dict{Symbol,
                                                                           Vector{Vector{Int64}}}
    time = petab_measurements.time
    imeasurements_t = Dict{Symbol, Vector{Vector{Int64}}}()
    for (id, ims) in imeasurements
        timepoints = time[ims]
        timepoints_unique = unique(sort(timepoints))
        itimepoints = fill(Int64[], length(timepoints_unique))
        for i in eachindex(itimepoints)
            itimepoints[i] = ims[findall(x -> x == timepoints_unique[i], timepoints)]
        end
        imeasurements_t[id] = itimepoints
    end
    return imeasurements_t
end

function _get_smatrixindices(experiment_ids::Vector{Symbol}, tsaves::Dict{Symbol, Vector{Float64}})::Dict{Symbol, UnitRange{Int64}}
    smatrixindices = Dict{Symbol, UnitRange{Int64}}()
    istart, iend = 1, 0
    for cid in experiment_ids
        iend += length(tsaves[cid])
        smatrixindices[cid] = istart:iend
        istart = iend + 1
    end
    return smatrixindices
end
