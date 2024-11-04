function _switch_condition(oprob::ODEProblem, cid::Symbol, xdynamic::AbstractVector,
                           xnn::Dict{Symbol, ComponentArray}, model_info::ModelInfo,
                           cache::PEtabODEProblemCache, nn_pre_ode::Dict{Symbol, Dict{Symbol, NNPreODE}};
                           sensitivites::Bool = false, simid::Union{Nothing, Symbol} = nothing)::ODEProblem
    @unpack xindices, model, nstates = model_info
    simid = isnothing(simid) ? cid : simid

    # Each simulation condition needs to have its own associated u0 and p vector, as these
    # vectors can be used in latter computations when computing the observables, hence
    # nothing is allowed to be over-written by another condition
    p = get_tmp(cache.pode[cid], oprob.p)
    u0 = get_tmp(cache.u0ode[cid], oprob.p)
    p .= oprob.p
    @views u0 .= oprob.u0[1:length(u0)]

    # Condition specific parameters
    map_cid = xindices.maps_conidition_id[simid]
    p[map_cid.isys_constant_values] .= map_cid.constant_values
    p[map_cid.ix_sys] .= xdynamic[map_cid.ix_dynamic]

    # Potential Neural-Network parameters (in this case p must be a ComponentArray)
    for (netid, xnet) in xnn
        !haskey(p, netid) && continue
        p[netid] .= xnet
    end

    # Parameters which are set by a neural net. TODO: Make function maybe
    if haskey(nn_pre_ode, simid)
        nns_pre_ode = nn_pre_ode[simid]
        maps_nns = xindices.maps_nn_pre_ode[simid]
        for (netid, nn_pre_ode) in nns_pre_ode
            map_nn = maps_nns[netid]
            # In case of neural nets being computed before the function call,
            # nn_pre_ode.outputs has already been assigned
            outputs = get_tmp(nn_pre_ode.outputs, p)
            if nn_pre_ode.computed[1] == false
                pnn = xnn[netid]
                _x = _get_nn_pre_ode_x(nn_pre_ode, xdynamic, pnn, map_nn)
                nn_pre_ode.nn!(outputs, _x)
            end
            p[map_nn.ioutput_sys] .= outputs
        end
    end

    # Initial state can depend on condition specific parameters
    model.u0!((@view u0[1:nstates]), p)

    _oprob = remake(oprob, p = p, u0 = u0)
    # In case we solve the forward sensitivity equations we must adjust the initial
    # sensitives by computing the jacobian at t0, and note we have larger than usual
    # u0 as it includes the sensitivites. Must come after the remake as the remake
    # resets u0 values for the sensitivites
    if sensitivites == true
        St0::Matrix{Float64} = zeros(Float64, nstates, length(p))
        ForwardDiff.jacobian!(St0, model.u0, p)
        _oprob.u0 .= vcat(u0, vec(St0))
    end
    return _oprob
end

function _get_tsave(save_observed_t::Bool, simulation_info::SimulationInfo, cid::Symbol,
                    ntimepoints_save::Integer)::Vector{Float64}
    tmax = simulation_info.tmaxs[cid]
    if save_observed_t == true
        return simulation_info.tsaves[cid]
    elseif ntimepoints_save > 0
        return collect(LinRange(0.0, tmax, ntimepoints_save))
    else
        return Float64[]
    end
end

function _is_dense(save_observed_t::Bool, dense_sol::Bool, ntimepoints_save::Integer)::Bool
    if save_observed_t == true
        return false
    elseif ntimepoints_save > 0
        return false
    else
        return dense_sol
    end
end

function _get_cbs(oprob::ODEProblem, simulation_info::SimulationInfo, cid::Symbol,
                  sensealg)::SciMLBase.DECallback
    return simulation_info.callbacks[cid]
end

function _get_tspan(oprob::ODEProblem, tmax::Float64, solver::SciMLAlgorithm,
                    float_tspan::Bool)::ODEProblem
    # When tmax=Inf and a multistep BDF Julia method, e.g. QNDF, is used tmax must be inf,
    # else if it is a large number such as 1e8 the dt_min is set to a large value making
    # the solver fail. Sundials solvers on the other hand are not compatible with
    # timespan = (0.0, Inf), hence for these we use timespan = (0.0, 1e8)
    # u0tmp needed as remake resets sensitivity initial values to zero
    u0tmp = oprob.u0 |> deepcopy
    tmax = _get_tmax(tmax, solver)
    if float_tspan == true
        _oprob = remake(oprob, tspan = (0.0, tmax))
    else
        _oprob = remake(oprob, tspan = convert.(eltype(oprob.p), (0.0, tmax)))
    end
    _oprob.u0 .= u0tmp
    return _oprob
end

function _get_tmax(tmax::Float64, solver::Union{CVODE_BDF, CVODE_Adams})::Float64
    return isinf(tmax) ? 1e8 : tmax
end
function _get_tmax(tmax::Float64, solver::Union{Vector{Symbol}, SciMLAlgorithm})::Float64
    return tmax
end

function _get_preeq_ids(simulation_info::SimulationInfo,
                        cids::Vector{Symbol})::Vector{Symbol}
    if cids[1] == :all
        return unique(simulation_info.conditionids[:pre_equilibration])
    else
        which_id = findall(x -> x in simulation_info.conditionids[:experiment], cids)
        return unique(simulation_info.conditionids[:pre_equilibration][which_id])
    end
end

function _set_cond_const_parameters!(p::AbstractVector, xdynamic::AbstractVector,
                                     xindices::ParameterIndices)::Nothing
    map_oprob = xindices.map_odeproblem
    @views p[map_oprob.sys_to_dynamic] .= xdynamic[map_oprob.dynamic_to_sys]
    return nothing
end
