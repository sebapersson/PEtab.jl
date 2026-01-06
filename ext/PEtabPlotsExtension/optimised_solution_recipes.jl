const ALLOWED_SOLUTION_PLOTS = [
    :model_fit,
    :residuals,
    :standardized_residuals
]

# Plots the optimized solution, and compares it to the data. Either by directly plotting
# the model fit, or by plotting the residuals
@recipe function f(res::PEtab.EstimationResult, prob::PEtabODEProblem;
                   plot_type = :model_fit, observable_ids = nothing, condition = nothing,
                   obsid_label = false)
    model_info = prob.model_info

    if !in(plot_type, ALLOWED_SOLUTION_PLOTS)
        error("Argument plot_type have an unrecognized value ($(plot_type)). Allowed \
               values are: $(ALLOWED_SOLUTION_PLOTS).")
    end

    observables_df = prob.model_info.model.petab_tables[:observables]
    if isnothing(observable_ids)
        observable_ids = observables_df[!, :observableId]
    else
        observable_ids = string.(observable_ids)
    end

    simulation_id = PEtab._get_simulation_id(condition, model_info)
    pre_equilibration_id = PEtab._get_pre_equilibration_id(condition, model_info)
    PEtab._check_condition_ids(simulation_id, pre_equilibration_id, model_info)

    xmin = PEtab._get_x(res)

    # Get plot options
    if isnothing(pre_equilibration_id)
        title --> "Condition: $(simulation_id)"
    else
        title --> "Condition: $(pre_equilibration_id) => $(simulation_id)"
    end

    if plot_type == :model_fit
        plot_info = _plot_model_fit(xmin, prob, simulation_id, pre_equilibration_id, observable_ids, obsid_label)
    else
        plot_info = _plot_residuals(xmin, prob, simulation_id, pre_equilibration_id, observable_ids, plot_type, obsid_label)
    end

    seriestype --> plot_info.seriestype
    color --> plot_info.color
    label --> plot_info.label
    xlabel --> "Time"
    ylabel --> plot_info.y_label
    return plot_info.x, plot_info.y
end

function PEtab.get_obs_comparison_plots(res::PEtab.EstimationResult, prob::PEtabODEProblem;
                                        kwargs...)
    comparison_dict = Dict()
    conditions_ids = prob.model_info.simulation_info.conditionids
    simulation_ids = string.(conditions_ids[:simulation])
    observable_ids = prob.model_info.model.petab_tables[:observables][!, :observableId]
    for (i, simulation_id) in pairs(simulation_ids)
        pre_equilibration_id = conditions_ids[:pre_equilibration][i]
        if pre_equilibration_id == :None
            condition = simulation_id
            experiment_id = simulation_id
        else
            condition = Symbol(pre_equilibration_id) => Symbol(simulation_id)
            experiment_id = "$(pre_equilibration_id)=>$(simulation_id)"
        end
        comparison_dict[experiment_id] = Dict()
        for obsid in observable_ids
            comparison_dict[experiment_id][obsid] = plot(res, prob; observable_ids = [obsid], condition = condition, kwargs...)
        end
    end
    return comparison_dict
end

function _plot_model_fit(xmin, prob, simulation_id, pre_equilibration_id, observable_ids, obsid_label)
    observables_df = prob.model_info.model.petab_tables[:observables]

    # Prepares empty vectors with required plot inputs.
    _seriestype = []
    _color = []
    _label = []
    x_vals = []
    y_vals = []

    # Loops through all observables, computing the required plot inputs.
    for (obs_idx, obs_id) in enumerate(observable_ids)
        model_fits = _get_observable(xmin, prob, simulation_id, pre_equilibration_id, obs_id)
        # Plot args.
        append!(_seriestype, [:scatter, :line])
        append!(_color, [obs_idx, obs_idx])
        iobs = findfirst(x -> x == obs_id, observables_df[!, :observableId])
        if obsid_label == false
            obs_formula = observables_df[iobs, :observableFormula]
            append!(_label, ["$(obs_formula) ($type)" for type in ["measured", "fitted"]])
        else
            obs_formula = observables_df[iobs, :observableId]
            append!(_label, ["$(obs_formula) ($type)" for type in ["measured", "fitted"]])
        end

        # Measured plot values.
        push!(x_vals, model_fits.t_obs)
        push!(y_vals, model_fits.h_obs)
        # Fitted plot values.
        push!(x_vals, model_fits.t_mod)
        push!(y_vals, model_fits.h_mod)
    end

    # Set reshaped plot arguments
    n_obs = length(observable_ids)
    _seriestype = reshape(_seriestype, 1, 2n_obs)
    _color = reshape(_color, 1, 2n_obs)
    _label = reshape(_label, 1, 2n_obs)
    return (x = x_vals, y = y_vals, color = _color, label = _label,
            seriestype = _seriestype, y_label ="Model and observed values")
end

function _plot_residuals(xmin, prob, simulation_id, pre_equilibration_id, observable_ids, plot_type, obsid_label)
    observables_df = prob.model_info.model.petab_tables[:observables]

    if plot_type == :residuals
        _y_label = "Residuals (simulated output - data)"
    elseif plot_type == :standardized_residuals
        _y_label = "Standardized residuals"
    end
    petab_measurements = prob.model_info.petab_measurements

    _seriestype = []
    _color = []
    _label = []
    x_vals = []
    y_vals = []

    # Loops through all observables, computing the required plot inputs.
    for (obs_idx, obs_id) in enumerate(observable_ids)
        idata = _get_index_data(simulation_id, pre_equilibration_id, obs_id, prob.model_info)
        isempty(idata) && continue

        t_observed = petab_measurements.time[idata]
        if plot_type == :residuals
            measurements_transformed = petab_measurements.measurements_transformed[idata]
            simulated_values = prob.simulated_values(xmin)[idata]
            residuals = (simulated_values - measurements_transformed)
        else
            residuals = prob.residuals(xmin)[idata]
        end

        iobs = findfirst(x -> x == obs_id, observables_df[!, :observableId])
        if obsid_label == false
            __label = observables_df[iobs, :observableFormula]
        else
            __label = observables_df[iobs, :observableId]
        end

        push!(x_vals, t_observed)
        push!(y_vals, residuals)
        append!(_seriestype, [:scatter])
        append!(_color, [obs_idx])
        append!(_label, ["$(__label)"])
    end

    # Set reshaped plot arguments
    n_obs = length(observable_ids)
    _seriestype = reshape(_seriestype, 1, n_obs)
    _color = reshape(_color, 1, n_obs)
    _label = reshape(_label, 1, n_obs)
    return (x = x_vals, y = y_vals, color = _color, label = _label,
            seriestype = _seriestype, y_label = _y_label)
end

"""
    _get_observable(x, prob::PEtabODEProblem, simulation_id, obsid)

Return the model values for a given obsid (observable id), simulation_id (condition id)
and pre_equilibration_id (pre-equilibration id) using the parameter vector x.

This function is primarily used for plotting purposes.

# Returns
- `t_observed::Vector{Float64}`: Time points for the observed data (x-axis).
- `h_observed::Vector{Float64}`: Observed data values corresponding to these time points.
- `t_model::Vector{Float64}`: Time points for the model data (x-axis).
- `h_model::Vector{Float64}`: Model's predicted values corresponding to these time points.
"""
function _get_observable(x, prob::PEtabODEProblem, simulation_id::Symbol, pre_equilibration_id::Union{Symbol, Nothing},
                         obsid::String)
    @unpack model_info, probinfo = prob
    measurements_df = model_info.model.petab_tables[:measurements]

    idata = _get_index_data(simulation_id, pre_equilibration_id, obsid, model_info)
    if isempty(idata)
        return (t_obs=Float64[], h_obs=Float64[], t_mod=Float64[], h_mod=Float64[])
    end
    t_observed = measurements_df[idata, :time]
    h_observed = measurements_df[idata, :measurement]

    # If we have observable parameters that scale the observable functions it does not
    # make sense to return a smooth ODESolution. If no such parameters are present, a
    # smooth function can be returned
    t_model = Float64[]
    h_model = Float64[]
    _x = collect(x)
    if isnothing(pre_equilibration_id)
        sol = PEtab.get_odesol(_x, prob; condition = simulation_id)
    else
        sol = PEtab.get_odesol(_x, prob; condition = pre_equilibration_id => simulation_id)
    end
    map_xobservables = model_info.xindices.xobservable_maps[idata]
    smooth_sol = all([map.nparameters == 0 for map in map_xobservables])

    # For smooth trajectory must solve ODE and compute the observable function
    if smooth_sol == true
        PEtab.split_x!(x, model_info.xindices, probinfo.cache)
        @unpack xobservable, xnondynamic = probinfo.cache
        xobservable_ps = PEtab.transform_x(xobservable, model_info.xindices,
                                           :xobservable, probinfo.cache)
        xnondynamic_ps = PEtab.transform_x(xnondynamic, model_info.xindices,
                                           :xnondynamic, probinfo.cache)
        for (i, t) in pairs(sol.t)
            u = sol[:, i]
            h = PEtab._h(u, t, sol.prob.p, xobservable_ps, xnondynamic_ps,
                         model_info.model, map_xobservables[1],
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

    return (t_obs=t_observed, h_obs=h_observed, t_mod=t_model, h_mod=h_model)
end

function _get_index_data(simulation_id, pre_equilibration_id, obsid, model_info)
    measurements_df = model_info.model.petab_tables[:measurements]
    simulation_ids = measurements_df[!, :simulationConditionId]
    observable_ids = measurements_df[!, :observableId]

    # Identify which data-points in measurement data to plot
    if isnothing(pre_equilibration_id)
        idata = findall(simulation_ids .== string(simulation_id) .&& observable_ids .== obsid)
    else
        pre_equilibration_ids = measurements_df[!, :preequilibrationConditionId]
        idata = findall(simulation_ids .== string(simulation_id) .&& observable_ids .== obsid .&& string(pre_equilibration_id) .== pre_equilibration_ids)
    end
    return idata
end
