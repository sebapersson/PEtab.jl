const NN_FUNCTION_PLOTS = [
    :best_function,
    :function_ensemble,
]
const ALLOWED_SOLUTION_PLOTS = [
    :model_fit,
    :residuals,
    :standardized_residuals,
    NN_FUNCTION_PLOTS...,
]

# Plots the optimized solution, and compares it to the data. Either by directly plotting
# the model fit, or by plotting the residuals
Plots.@recipe function f(
        res::PEtab.EstimationResult, prob::PEtabODEProblem; plot_type = :model_fit,
        observable_ids = nothing, condition = nothing, observable_id_label = false,
        experiment = nothing,
        # Relevant for fitted neural network plotting only.
        nn_idx = 1, x_support = nothing, num_plotted_nn = nothing, loss_thres = Inf,
        plt_dens = 200, plotted_dim = 1, clustering_function = objective_value_clustering
    )
    model_info = prob.model_info
    if !in(plot_type, ALLOWED_SOLUTION_PLOTS)
        error("Argument plot_type have an unrecognized value ($(plot_type)). Allowed \
               values are: $(ALLOWED_SOLUTION_PLOTS).")
    end

    if plot_type in NN_FUNCTION_PLOTS
        # For plotting fitted function. Implemented in the "ude_functions_recipes.jl" file.
        xlabel, title, plot_info = _plot_ude_function_fit(
            res, prob, plot_type, nn_idx,
            x_support, num_plotted_nn, loss_thres, plt_dens, plotted_dim,
            clustering_function
        )
    else
        observables_df = prob.model_info.model.petab_tables[:observables]
        if isnothing(observable_ids)
            observable_ids = observables_df[!, :observableId]
        else
            observable_ids = string.(observable_ids)
        end

        # Get plot options
        PEtab._check_experiment_id(condition, experiment, model_info)
        simulation_id = PEtab._get_simulation_id(condition, experiment, model_info)
        pre_equilibration_id = PEtab._get_pre_equilibration_id(
            condition, experiment, model_info
        )
        PEtab._check_condition_ids(simulation_id, pre_equilibration_id, model_info)
        petab_version = PEtab._get_version(model_info)

        if petab_version == "2.0.0" && isnothing(experiment)
            experiment_id = first(split("$simulation_id", '_'))
            title = "Experiment: $(experiment_id)"
        elseif petab_version == "2.0.0" && !isnothing(experiment)
            title = "Experiment: $(experiment)"
        elseif isnothing(pre_equilibration_id)
            title = "Condition: $(simulation_id)"
        else
            title = "Condition: $(pre_equilibration_id) => $(simulation_id)"
        end

        xmin = PEtab._get_x(res)
        if plot_type == :model_fit
            plot_info = _plot_model_fit(
                xmin, prob, condition, experiment, observable_ids, observable_id_label
            )
        else
            plot_info = _plot_residuals(
                xmin, prob, condition, experiment, observable_ids, plot_type,
                observable_id_label
            )
        end

        xlabel = "Time"
    end

    seriestype --> plot_info.seriestype
    color --> plot_info.color
    label --> plot_info.label
    xlabel --> xlabel
    ylabel --> plot_info.y_label
    title --> title
    return plot_info.x, plot_info.y
end

function PEtab.get_obs_comparison_plots(
        res::PEtab.EstimationResult, prob::PEtabODEProblem; kwargs...
    )
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
        for observable_id in observable_ids
            comparison_dict[experiment_id][observable_id] = Plots.plot(
                res, prob; observable_ids = [observable_id], condition = condition,
                kwargs...
            )
        end
    end
    return comparison_dict
end

function _plot_model_fit(
        xmin, prob, condition, experiment, observable_ids, observable_id_label
    )
    observables_df = prob.model_info.model.petab_tables[:observables]

    # Prepares empty vectors with required plot inputs.
    _seriestype = []
    _color = []
    _label = []
    x_vals = []
    y_vals = []

    # Loops through all observables, computing the required plot inputs.
    for (obs_idx, obs_id) in enumerate(observable_ids)
        model_fits = PEtab._get_observable(xmin, prob, condition, experiment, obs_id)
        # Plot args.
        append!(_seriestype, [:scatter, :line])
        append!(_color, [obs_idx, obs_idx])
        iobs = findfirst(x -> x == obs_id, observables_df[!, :observableId])
        if observable_id_label == false
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
    return (
        x = x_vals, y = y_vals, color = _color, label = _label,
        seriestype = _seriestype, y_label = "Model and observed values",
    )
end

function _plot_residuals(
        xmin, prob, condition, experiment, observable_ids, plot_type, observable_id_label
    )
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
        idata = PEtab._get_index_data(condition, experiment, obs_id, prob.model_info)
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
        if observable_id_label == false
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
    return (
        x = x_vals, y = y_vals, color = _color, label = _label,
        seriestype = _seriestype, y_label = _y_label,
    )
end
