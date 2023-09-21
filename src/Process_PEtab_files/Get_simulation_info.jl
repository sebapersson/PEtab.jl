#=
    Functions for processing the PEtab observables and measurements file into a Julia
    SimulationInfo struct, which contains information required to carry out forward ODE simulations
    (condition-id), and indices for mapping simulations to measurement data.
=#


function process_simulationinfo(petab_model::PEtabModel,
                                measurement_info::MeasurementsInfo;
                                sensealg)::SimulationInfo

    # An experimental Id is uniqely defined by a Pre-equlibrium- and Simulation-Id, where the former can be
    # empty. For each experimental ID we store three indices, i) preEqulibriumId, ii) simulationId and iii)
    # experimentalId (concatenation of two).
    pre_equilibration_condition_id::Vector{Symbol} = Vector{Symbol}(undef, 0)
    simulation_condition_id::Vector{Symbol} = Vector{Symbol}(undef, 0)
    experimental_condition_id::Vector{Symbol} = Vector{Symbol}(undef, 0)
    for i in eachindex(measurement_info.pre_equilibration_condition_id)
        # In case model has steady-state simulations prior to matching against data
        _pre_equilibration_condition_id = measurement_info.pre_equilibration_condition_id[i]
        if _pre_equilibration_condition_id == :None
            measurement_info.simulation_condition_id[i] ∈ experimental_condition_id && continue
            pre_equilibration_condition_id = vcat(pre_equilibration_condition_id, :None)
            simulation_condition_id = vcat(simulation_condition_id, measurement_info.simulation_condition_id[i])
            experimental_condition_id = vcat(experimental_condition_id, measurement_info.simulation_condition_id[i])
            continue
        end

        # For cases with no steady-state simulations
        _experimental_condition_id = Symbol(string(measurement_info.pre_equilibration_condition_id[i]) * string(measurement_info.simulation_condition_id[i]))
        if _experimental_condition_id ∉ experimental_condition_id
            pre_equilibration_condition_id = vcat(pre_equilibration_condition_id, measurement_info.pre_equilibration_condition_id[i])
            simulation_condition_id = vcat(simulation_condition_id, measurement_info.simulation_condition_id[i])
            experimental_condition_id = vcat(experimental_condition_id, _experimental_condition_id)
            continue
        end
    end

    has_pre_equilibration_condition_id::Bool = all(pre_equilibration_condition_id.== :None) ? false : true

    # When computing the gradient and hessian the ODE-system needs to be resolved to compute the gradient
    # of the dynamic parameters, while for the observable/sd parameters the system should not be resolved.
    # Hence we need a specific dictionary with ODE solutions when compuating derivatives.
    ode_sols::Dict{Symbol, Union{Nothing, ODESolution}} = Dict{Symbol, ODESolution}()
    ode_sols_pre_equlibrium::Dict{Symbol, Union{Nothing, ODESolution, SciMLBase.NonlinearSolution}} = Dict{Symbol, ODESolution}()
    ode_sols_derivatives::Dict{Symbol, Union{Nothing, ODESolution}} = Dict{Symbol, ODESolution}()
    for i in eachindex(experimental_condition_id)
        ode_sols[experimental_condition_id[i]] = nothing
        ode_sols_derivatives[experimental_condition_id[i]] = nothing
        if pre_equilibration_condition_id[i] != :None && pre_equilibration_condition_id[i] ∉ keys(ode_sols_pre_equlibrium)
            ode_sols_pre_equlibrium[pre_equilibration_condition_id[i]] = nothing
        end
    end

    # Precompute the max simulation time for each experimental_condition_id
    _tmax = Tuple(compute_tmax(pre_equilibration_condition_id[i], simulation_condition_id[i], measurement_info) for i in eachindex(pre_equilibration_condition_id))
    tmax::Dict{Symbol, Float64} = Dict([experimental_condition_id[i] => _tmax[i] for i in eachindex(_tmax)])

    # Precompute which time-points we have observed data at for experimental_condition_id (used in saveat for ODE solution)
    _time_observed = Tuple(compute_time_observed(pre_equilibration_condition_id[i], simulation_condition_id[i], measurement_info) for i in eachindex(pre_equilibration_condition_id))
    time_observed::Dict{Symbol, Vector{Float64}} = Dict([experimental_condition_id[i] => _time_observed[i] for i in eachindex(_time_observed)])

    # Precompute indices in measurement_info (i_measurement) for each experimental_condition_id
    _i_measurements_observed = Tuple(_compute_time_indices(pre_equilibration_condition_id[i], simulation_condition_id[i], measurement_info) for i in eachindex(pre_equilibration_condition_id))
    i_measurements_observed::Dict{Symbol, Vector{Int64}} = Dict([experimental_condition_id[i] => _i_measurements_observed[i] for i in eachindex(_i_measurements_observed)])

    # Precompute for each measurement (entry in i_measurement) a vector which holds the corresponding index in ode_sol.t
    # accounting for experimental_condition_id
    i_time_ode_sol::Vector{Int64} = compute_time_index_ode_sol(pre_equilibration_condition_id, simulation_condition_id, measurement_info)

    # When computing the gradients via forward sensitivity equations we need to track where, in the concatanated
    # ode_sol.t (accross all condition) the time-points for an experimental conditions start as we can only
    # compute the sensitivity matrix accross all conditions.
    time_position_ode_sol::Dict{Symbol, UnitRange{Int64}} = get_time_position_ode_sol(experimental_condition_id, time_observed)

    # Precompute a vector of vector where vec[i] gives the indices for time-point ti in measurement_info for an
    # experimental_condition_id. Needed for the lower level adjoint interface where we must track the number of
    # repats per time-point (when using dgdu_discrete and dgdp_discrete)
    _i_per_time_point = Tuple(compute_time_indices(pre_equilibration_condition_id[i], simulation_condition_id[i], measurement_info) for i in eachindex(pre_equilibration_condition_id))
    i_per_time_point::Dict{Symbol, Vector{Vector{Int64}}} = Dict([(experimental_condition_id[i], _i_per_time_point[i]) for i in eachindex(_i_per_time_point)])

    # Some models, e.g those with time dependent piecewise statements, have callbacks encoded. When doing adjoint
    # sensitivity analysis we need to track these callbacks, hence they must be stored in simulation_info.
    callbacks = Dict{Symbol, SciMLBase.DECallback}()
    tracked_callbacks = Dict{Symbol, SciMLBase.DECallback}()
    for name in experimental_condition_id
        callbacks[name] = deepcopy(petab_model.model_callbacks)
    end

    simulation_info = SimulationInfo(pre_equilibration_condition_id,
                                     simulation_condition_id,
                                     experimental_condition_id,
                                     has_pre_equilibration_condition_id,
                                     ode_sols,
                                     ode_sols_derivatives,
                                     ode_sols_pre_equlibrium,
                                     [true], # Bool tracking if we could solve ODE 
                                     tmax,
                                     time_observed,
                                     i_measurements_observed,
                                     i_time_ode_sol,
                                     i_per_time_point,
                                     time_position_ode_sol,
                                     callbacks,
                                     tracked_callbacks,
                                     sensealg)
    return simulation_info
end


function _compute_time_indices(pre_equilibration_condition_id::Symbol, simulation_condition_id::Symbol, measurement_info::MeasurementsInfo)::Vector{Int64}
    i_timepoints = findall(i -> pre_equilibration_condition_id == measurement_info.pre_equilibration_condition_id[i] && simulation_condition_id == measurement_info.simulation_condition_id[i], eachindex(measurement_info.time))
    return i_timepoints
end


function compute_time_indices(pre_equilibration_condition_id::Symbol, simulation_condition_id::Symbol, measurement_info::MeasurementsInfo)::Vector{Vector{Int64}}
    _i_timepoints = _compute_time_indices(pre_equilibration_condition_id, simulation_condition_id, measurement_info)
    timepoints = measurement_info.time[_i_timepoints]
    timepoints_unique = sort(unique(timepoints))
    i_timepoints = Vector{Vector{Int64}}(undef, length(timepoints_unique))
    for i in eachindex(i_timepoints)
        i_timepoints[i] = _i_timepoints[findall(x -> x == timepoints_unique[i], timepoints)]
    end

    return i_timepoints
end


function compute_tmax(pre_equilibration_condition_id::Symbol, simulation_condition_id::Symbol, measurement_info::MeasurementsInfo)::Float64
    i_timepoints = _compute_time_indices(pre_equilibration_condition_id, simulation_condition_id, measurement_info)
    return Float64(maximum(measurement_info.time[i_timepoints]))
end


function compute_time_observed(pre_equilibration_condition_id::Symbol, simulation_condition_id::Symbol, measurement_info::MeasurementsInfo)::Vector{Float64}
    i_timepoints = _compute_time_indices(pre_equilibration_condition_id, simulation_condition_id, measurement_info)
    return sort(unique(measurement_info.time[i_timepoints]))
end


# For each experimental condition (forward ODE-solution) compute index in ode_sol.t for any index
# i_measurement in measurement_info.time[i_measurement]
function compute_time_index_ode_sol(pre_equilibration_condition_id::Vector{Symbol},
                                    simulation_condition_id::Vector{Symbol},
                                    measurement_info::MeasurementsInfo)::Vector{Int64}

    i_time_ode_sol::Vector{Int64} = Vector{Int64}(undef, length(measurement_info.time))
    for i in eachindex(simulation_condition_id)
        i_timepoints = _compute_time_indices(pre_equilibration_condition_id[i], simulation_condition_id[i], measurement_info)
        timepoints = measurement_info.time[i_timepoints]
        timepoints_unique = sort(unique(timepoints))
        for iT in i_timepoints
            t = measurement_info.time[iT]
            i_time_ode_sol[iT] = findfirst(x -> x == t, timepoints_unique)
        end
    end
    return i_time_ode_sol
end


# For each time-point in the concatanated ode_sol.t (accross all conditions) get which index it corresponds
# to in the concatanted time_observed (accross all conditions). This is needed when computing forward sensitivites
# via forward mode automatic differentiation because here we get a big sensitivity matrix accross all experimental
# conditions, where S[i:(i+nStates)] row corresponds to the sensitivites at a specific time-point.
# An assumption made here is that we solve the ODE:s in the order of experimental_condition_id (which is true)
function get_time_position_ode_sol(experimental_condition_id::Vector{Symbol},
                                       time_observed::Dict)::Dict{Symbol, UnitRange{Int64}}

    i_start::Int64 = 1
    position_ode_sol::Dict{Symbol, UnitRange{Int64}} = Dict{Symbol, UnitRange{Int64}}()
    for i in eachindex(experimental_condition_id)
        time_observed_condition = time_observed[experimental_condition_id[i]]
        _position_ode_sol_condition = i_start:(i_start-1+length(time_observed_condition))
        i_start = _position_ode_sol_condition[end] + 1
        position_ode_sol[experimental_condition_id[i]] = _position_ode_sol_condition
    end

    return position_ode_sol
end
