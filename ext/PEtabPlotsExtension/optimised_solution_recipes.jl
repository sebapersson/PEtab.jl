# Plots the optimised solution, and compares it to the data.
@recipe function f(res::Union{PEtabOptimisationResult, PEtabMultistartResult},
                   prob::PEtabODEProblem; obsids = nothing, cid = nothing)
    observables_df = prob.model_info.model.petab_tables[:observables]
    if isnothing(obsids)
        obsids = observables_df[!, :observableId]
    end
    if isnothing(cid)
        cid = prob.model_info.model.petab_tables[:conditions][!, :conditionId][1]
    end

    # Get plot options
    title --> cid
    ylabel --> "Model and observed values"
    xlabel --> "Time"

    # Prepares empty vectors with required plot inputs.
    seriestype = []
    color = []
    label = []
    x_vals = []
    y_vals = []

    # Loops through all observables, computing the required plot inputs.
    for (obs_idx, obs_id) in enumerate(obsids)
        t_observed, h_observed, label_observed, t_model, h_model, label_model = _get_observable(res.xmin,
                                                                                                prob,
                                                                                                cid,
                                                                                                obs_id)

        # Plot args.
        append!(seriestype, [:scatter, :line])
        append!(color, [obs_idx, obs_idx])
        iobs = findfirst(x -> x == obs_id, observables_df[!, :observableId])
        obs_formula = observables_df[iobs, :observableFormula]
        append!(label, ["$(obs_formula) ($type)" for type in ["measured", "fitted"]])

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

function PEtab.get_obs_comparison_plots(res::Union{PEtabOptimisationResult,
                                                   PEtabMultistartResult},
                                        prob::PEtabODEProblem; kwargs...)
    comparison_dict = Dict()
    cids = prob.model_info.model.petab_tables[:conditions][!, :conditionId]
    obsids = prob.model_info.model.petab_tables[:observables][!, :observableId]
    for cid in cids
        comparison_dict[cid] = Dict()
        for obsid in obsids
            comparison_dict[cid][obsid] = plot(res, prob; obsids = [obsid], cid = cid,
                                               kwargs...)
        end
    end
    return comparison_dict
end

"""
    _get_observable(x, prob::PEtabODEProblem, cid::String, obsid::String)

Return the model values for a given obsid and cid (condition id) using
the parameter vector x.

This function is primarily used for plotting purposes. It is important to note that for a
single cid and obsid, the function may return simulation results
corresponding to multiple conditions if the model has different pre-equilibrium ids.

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
function _get_observable(x, prob::PEtabODEProblem, cid::String, obsid::String)
    @unpack model_info, probinfo = prob
    # Sanity check that a valid id has been provided
    measurements_df = model_info.model.petab_tables[:measurements]
    cids = measurements_df[!, :simulationConditionId]
    obsids = measurements_df[!, :observableId]
    @assert cid in cids "$cid in not one of the model's simulations ids"
    @assert obsid in obsids "$obsid in not one of the model's observable ids"

    # Identify which data-points in measurment data to plot
    idata = findall(cids .== cid .&& obsids .== obsid)
    if isempty(idata)
        return Float64[], Float64[], String[], Float64[], Float64[], String[]
    end

    # Extract measurement value and observed time for the data. The same condition id
    # can due to pre-equilibration correspond to different simulation scenario. This
    # must be tracked via label_observed
    preeq_ids = prob.model_info.petab_measurements.pre_equilibration_condition_id[idata]
    t_observed = measurements_df[idata, :time]
    h_observed = measurements_df[idata, :measurement]
    label_observed = fill("", length(idata))
    for i in eachindex(label_observed)
        if preeq_ids[i] == :None
            label_observed[i] = cid
        else
            label_observed[i] = "Pre_$(preeq_ids[i])_$(cid)"
        end
    end

    # Compute model values. If we have observable parameters that scale the observable
    # functions it does not make sense to return a smooth ODESolution, but if these are
    # not present it looks better with a smooth solution
    t_model = Float64[]
    h_model = Float64[]
    label_model = String[]
    _x = collect(x)
    for preeq_id in unique(preeq_ids)
        if preeq_id == :None
            _idata = idata
            sol = PEtab.get_odesol(_x, prob; cid = cid)
        else
            _idata = idata[findall(preeq_ids .== preeq_id)]
            sol = PEtab.get_odesol(_x, prob; cid = cid, preeq_id = preeq_id)
        end

        mapxobservables = model_info.xindices.mapxobservable[_idata]
        smooth_sol = all([map.nparameters == 0 for map in mapxobservables])
        # For smooth trajectory must solve ODE and compute the observable function
        if smooth_sol == true
            PEtab.split_x!(x, model_info.xindices, probinfo.cache)
            @unpack xobservable, xnondynamic = probinfo.cache
            xobservable_ps = PEtab.transform_x(xobservable, model_info.xindices,
                                               :xobservable, probinfo.cache)
            xnondynamic_ps = PEtab.transform_x(xnondynamic, model_info.xindices,
                                               :xnondynamic, probinfo.cache)
            _h_model = similar(sol.t)
            for (i, t) in pairs(sol.t)
                u = sol[:, i]
                h = PEtab._h(u, t, sol.prob.p, xobservable_ps, xnondynamic_ps,
                             model_info.model.h, mapxobservables[1],
                             Symbol(obsid), collect(prob.xnominal))
                push!(h_model, h)
            end
            t_model = vcat(t_model, sol.t)
            npoints = length(sol.t)
            # With observable parameters the simulated values can be used instead
        else
            t_model = vcat(t_model, measurements_df[_idata, :time])
            h_model = vcat(h_model, prob.simulated_values(x)[_idata])
            npoints = length(measurements_df[_idata, :time])
        end
        if preeq_id == :None
            _label = [cid for _ in 1:npoints]
        else
            _label = ["pre_$(preeq_id)_$(cid)" for _ in 1:npoints]
        end
        label_model = vcat(label_model, _label)
    end

    return t_observed, h_observed, label_observed, t_model, h_model, label_model
end
