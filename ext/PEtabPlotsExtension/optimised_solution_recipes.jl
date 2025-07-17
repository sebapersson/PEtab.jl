# Plots the optimized solution, and compares it to the data.
@recipe function f(res::PEtab.EstimationResult, prob::PEtabODEProblem; obsids = nothing,
                   cid = nothing, preeq_id = nothing, obsid_label::Bool = false)
    observables_df = prob.model_info.model.petab_tables[:observables]
    cid = isnothing(cid) ? cid : string(cid)
    preeq_id = isnothing(preeq_id) ? preeq_id : string(preeq_id)
    if isnothing(obsids)
        obsids = observables_df[!, :observableId]
    else
        obsids = string.(obsids)
    end
    condition_ids = prob.model_info.simulation_info.conditionids
    if isnothing(cid) && isnothing(preeq_id)
        cid = string(condition_ids[:simulation][1])
        preeq_id = string(condition_ids[:pre_equilibration][1])
    end
    if !isnothing(cid) && isnothing(preeq_id)
        i_cid = findall(x -> x == Symbol(cid), condition_ids[:simulation])
        if length(i_cid) > 1
            throw(ArgumentError("Simulation condition `$cid` appears in the measurement \
                table with multiple pre-equilibration condition IDs. When plotting model \
                fit, only one experimental condition (i.e., one pre-equilibration + \
                simulation condition pair) can be plotted. To fix this error, explicitly \
                provide both `cid` and `preeq_id` in the plot function call. For example:
                `plot(res, petab_prob; cid = :cond1, preeq_id = :cond_pre)`"))
        end
        preeq_id = string(condition_ids[:pre_equilibration][i_cid[1]])
    end

    # Get plot options
    if (isnothing(preeq_id) || preeq_id == "None")
        title --> cid
    else
        title --> "Pre-equilibration: $(preeq_id), Main: $(cid)"
    end
    ylabel --> "Model and observed values"
    xlabel --> "Time"

    # Prepares empty vectors with required plot inputs.
    seriestype = []
    color = []
    label = []
    x_vals = []
    y_vals = []

    # Loops through all observables, computing the required plot inputs.
    if res isa Union{AbstractVector, ComponentArray}
        xmin = res
    else
        xmin = res.xmin
    end
    for (obs_idx, obs_id) in enumerate(obsids)
        (t_observed, h_observed, t_model, h_model) = _get_observable(xmin, prob, cid,
                                                                     preeq_id, obs_id)

        # Plot args.
        append!(seriestype, [:scatter, :line])
        append!(color, [obs_idx, obs_idx])
        iobs = findfirst(x -> x == obs_id, observables_df[!, :observableId])
        if obsid_label == false
            obs_formula = observables_df[iobs, :observableFormula]
            append!(label, ["$(obs_formula) ($type)" for type in ["measured", "fitted"]])
        else
            obs_formula = observables_df[iobs, :observableId]
            append!(label, ["$(obs_formula) ($type)" for type in ["measured", "fitted"]])
        end

        # Measured plot values.
        push!(x_vals, t_observed)
        push!(y_vals, h_observed)

        # Fitted plot values.
        push!(x_vals, t_model)
        push!(y_vals, h_model)
    end

    # Set reshaped plot arguments
    n_obs = length(obsids)
    seriestype --> reshape(seriestype, 1, 2n_obs)
    color --> reshape(color, 1, 2n_obs)
    label --> reshape(label, 1, 2n_obs)

    # Return output.
    x_vals, y_vals
end

function PEtab.get_obs_comparison_plots(res::PEtab.EstimationResult, prob::PEtabODEProblem;
                                        kwargs...)
    comparison_dict = Dict()
    conditions_ids = prob.model_info.simulation_info.conditionids
    cids = string.(conditions_ids[:simulation])
    obsids = prob.model_info.model.petab_tables[:observables][!, :observableId]
    for (i, cid) in pairs(cids)
        preeq_id = conditions_ids[:pre_equilibration][i]
        if preeq_id == :None
            exp_id = cid
        else
            exp_id = "pre_$(preeq_id)_main_$(cid)"
        end
        comparison_dict[exp_id] = Dict()
        for obsid in obsids
            comparison_dict[exp_id][obsid] = plot(res, prob; obsids = [obsid], cid = cid,
                                                  preeq_id = preeq_id, kwargs...)
        end
    end
    return comparison_dict
end

"""
    _get_observable(x, prob::PEtabODEProblem, cid::String, obsid::String)

Return the model values for a given obsid (observable id), cid (condition id)
and preeq_id (pre-equilibration id) using the parameter vector x.

This function is primarily used for plotting purposes.

# Returns
- `t_observed::Vector{Float64}`: Time points for the observed data (x-axis).
- `h_observed::Vector{Float64}`: Observed data values corresponding to these time points.
- `label_observed::Vector{String}`: Labels denoting the condition id, considering any pre-equilibrium scenarios.
- `t_model::Vector{Float64}`: Time points for the model data (x-axis).
- `h_model::Vector{Float64}`: Model's predicted values corresponding to these time points.
- `label_model::Vector{String}`: Model labels denoting the condition id, considering any pre-equilibrium scenarios.
- `smooth_sol::Bool`: Indicates whether the returned solution is smooth, i.e., there are no
    observable parameters.
"""
function _get_observable(x, prob::PEtabODEProblem, cid::String, preeq_id::String,
                         obsid::String)
    @unpack model_info, probinfo = prob
    # Sanity check that a valid id has been provided
    measurements_df = model_info.model.petab_tables[:measurements]
    cids = measurements_df[!, :simulationConditionId]
    obsids = measurements_df[!, :observableId]
    @assert cid in cids "$cid in not one of the model's simulations ids"
    @assert obsid in obsids "$obsid in not one of the model's observable ids"

    # Identify which data-points in measurement data to plot
    if preeq_id == "None"
        idata = findall(cids .== cid .&& obsids .== obsid)
    else
        preeq_ids = measurements_df[!, :preequilibrationConditionId]
        @assert preeq_id in preeq_ids "$preeq_id in not one of the model's \
            pre-equilibration ids"
        idata = findall(cids .== cid .&& obsids .== obsid .&& preeq_id .== preeq_ids)
    end
    if isempty(idata)
        return Float64[], Float64[], String[], Float64[], Float64[], String[]
    end

    # Extract measurement value, observed time and labels for the measurement data
    preeq_ids = prob.model_info.petab_measurements.pre_equilibration_condition_id[idata]
    t_observed = measurements_df[idata, :time]
    h_observed = measurements_df[idata, :measurement]

    # If we have observable parameters that scale the observable functions it does not
    # make sense to return a smooth ODESolution. If no such parameters are present, a
    # smooth function can be returned
    t_model = Float64[]
    h_model = Float64[]
    _x = collect(x)
    if preeq_id == "None"
        sol = PEtab.get_odesol(_x, prob; cid = cid)
    else
        sol = PEtab.get_odesol(_x, prob; cid = cid, preeq_id = preeq_id)
    end
    map_xobservables = model_info.xindices.mapxobservable[idata]
    smooth_sol = all([map.nparameters == 0 for map in map_xobservables])

    # For smooth trajectory must solve ODE and compute the observable function
    if smooth_sol == true
        PEtab.split_x!(x, model_info.xindices, probinfo.cache)
        @unpack xobservable, xnondynamic = probinfo.cache
        xobservable_ps = PEtab.transform_x(xobservable, model_info.xindices,
                                           :xobservable_mech, probinfo.cache)
        xnondynamic_ps = PEtab.transform_x(xnondynamic, model_info.xindices,
                                           :xnondynamic_mech, probinfo.cache)
        for (i, t) in pairs(sol.t)
            u = sol[:, i]
            h = PEtab._h(u, t, sol.prob.p, xobservable_ps, xnondynamic_ps,
                         model_info.model.h, map_xobservables[1],
                         Symbol(obsid), collect(prob.xnominal))
            push!(h_model, h)
        end
        t_model = vcat(t_model, sol.t)
        npoints = length(sol.t)
        # With observable parameters the simulated values can be used instead
    else
        t_model = vcat(t_model, measurements_df[idata, :time])
        h_model = vcat(h_model, prob.simulated_values(x)[idata])
        npoints = length(measurements_df[idata, :time])
    end

    return t_observed, h_observed, t_model, h_model
end
