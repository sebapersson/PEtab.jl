function SimulationInfo(model_callbacks::SciMLBase.DECallback,
                        measurement_info::MeasurementsInfo; sensealg)::SimulationInfo
    conditionids = _get_conditionids(measurement_info)
    has_pre_equilibration = !all(conditionids[:pre_equilibration] .== :None)

    # Precompute values needed by the ODE-solver, such as tmax, tsave...
    tmaxs = _get_tmaxs(conditionids, measurement_info)
    tsaves = _get_tsaves(conditionids, measurement_info)

    # Indicies for getting measurement points for each condition. The second is a vector
    # of vector that accounts for multiple measurements per time-point which needs to
    # be accounted for in gradient compuations
    # TODO: imeasurements is redundant
    imeasurements = _get_imeasurements(conditionids, measurement_info)
    imeasurements_t = _get_imeasurements_t(imeasurements, measurement_info)
    # Time indicies in ODESolution for each measurement
    imeasurements_t_sol = _get_imeasurements_t_sol(imeasurements, measurement_info)
    # When computing forward sensitivites via forward mode automatic differentiation we get a
    # big sensitivity matrix accross all experimental conditions, where S[i:(i+nStates)]
    # corresponds to the sensitivites at a specific time-point. Here the indicies for each
    # experimental condition is computed. Note, this assumes that we solve the ODE:s in the
    # order of experimental_id (which is true)
    smatrixindices = _get_smatrixindices(conditionids[:experiment], tsaves)

    # When computing the gradient/Hessian for parameter not in the ODE-system with autodiff,
    # we do not need to solve (trace) the ODE solution. We only need an ODESolution.
    # Therefore the ODESolutions need to be stored in a Dict
    odesols = Dict{Symbol, ODESolution}()
    odesols_preeq = Dict{Symbol, Union{ODESolution, SciMLBase.NonlinearSolution}}()
    odesols_derivative = Dict{Symbol, ODESolution}()

    # Some models, e.g those with time dependent piecewise statements, have callbacks
    # encoded. For adjoint sensitivity analysis these most be tracked
    # sensitivity analysis we need to track these callbacks, hence they must be stored in simulation_info.
    callbacks = Dict{Symbol, SciMLBase.DECallback}()
    tracked_callbacks = Dict{Symbol, SciMLBase.DECallback}()
    for id in conditionids[:experiment]
        callbacks[id] = deepcopy(model_callbacks)
    end

    could_solve = [true]
    return SimulationInfo(conditionids, has_pre_equilibration, tmaxs, tsaves, imeasurements,
                          imeasurements_t, imeasurements_t_sol, smatrixindices, odesols,
                          odesols_derivative, odesols_preeq, could_solve, callbacks,
                          tracked_callbacks, sensealg)
end

function _get_conditionids(measurement_info::MeasurementsInfo)::Dict{Symbol, Vector{Symbol}}
    @unpack pre_equilibration_condition_id, simulation_condition_id = measurement_info
    # An experimental id is uniqely defined by a pre-equlibrium and simulation id, where
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

function _get_tsaves(conditionids::Dict{Symbol, Vector{Symbol}},
                     measurement_info::MeasurementsInfo)::Dict{Symbol, Vector{Float64}}
    tsave = Dict{Symbol, Vector{Float64}}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        preeqids, cids = conditionids[:pre_equilibration][i], conditionids[:simulation][i]
        it = _get_tindices(preeqids, cids, measurement_info)
        tsave[experiment_id] = sort(unique(measurement_info.time[it]))
    end
    return tsave
end

function _get_tmaxs(conditionids::Dict{Symbol, Vector{Symbol}},
                    measurement_info::MeasurementsInfo)::Dict{Symbol, Float64}
    tmaxs = Dict{Symbol, Float64}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        preeqids, cids = conditionids[:pre_equilibration][i], conditionids[:simulation][i]
        it = _get_tindices(preeqids, cids, measurement_info)
        tmaxs[experiment_id] = maximum(measurement_info.time[it])
    end
    return tmaxs
end

function _get_imeasurements(conditionids::Dict{Symbol, Vector{Symbol}},
                            measurement_info::MeasurementsInfo)::Dict{Symbol, Vector{Int64}}
    imeasurements = Dict{Symbol, Vector{Int64}}()
    for (i, experiment_id) in pairs(conditionids[:experiment])
        preeqids, cids = conditionids[:pre_equilibration][i], conditionids[:simulation][i]
        it = _get_tindices(preeqids, cids, measurement_info)
        imeasurements[experiment_id] = it
    end
    return imeasurements
end

function _get_tindices(pre_equilibration_id::Symbol, simulation_id::Symbol,
                       measurement_info::MeasurementsInfo)::Vector{Int64}
    @unpack pre_equilibration_condition_id, simulation_condition_id = measurement_info
    ipreeq = findall(x -> x == pre_equilibration_id, pre_equilibration_condition_id)
    isim = findall(x -> x == simulation_id, simulation_condition_id)
    return intersect(ipreeq, isim)
end

function _get_imeasurements_t_sol(imeasurements::Dict{Symbol, Vector{Int64}},
                                  measurement_info::MeasurementsInfo)::Vector{Int64}
    time = measurement_info.time
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
                              measurement_info::MeasurementsInfo)::Dict{Symbol,
                                                                        Vector{Vector{Int64}}}
    time = measurement_info.time
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

function _get_smatrixindices(experiment_ids::Vector{Symbol},
                             tsaves::Dict{Symbol, Vector{Float64}})::Dict{Symbol,
                                                                          UnitRange{Int64}}
    istart = 1
    smatrixindices = Dict{Symbol, UnitRange{Int64}}()
    for experiment_id in experiment_ids
        tsave = tsaves[experiment_id]
        smatrix_index = istart:(istart - 1 + length(tsave)) |> UnitRange{Int64}
        smatrixindices[experiment_id] = smatrix_index
        istart = smatrix_index[end] + 1
    end
    return smatrixindices
end
