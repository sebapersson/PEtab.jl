# Master function for handling the plotting of fitted UDE functions. Handles both the
# `:best_function` and `:function_ensemble` cases.
function _plot_ude_function_fit(
        res::PEtab.EstimationResult, prob::PEtabODEProblem, plot_type,
        nn_idx, x_support, num_plotted_nn, loss_thres, plt_dens, plotted_dim, clustering_function
    )
    # Extract all the ML functional calls within the model.
    # Check that `nn_idx` is OK. Checks that the model have one input and one input.
    if prob.model_info.model.sys isa ODEProblem
        throw(
            ArgumentError(
                "Plotting of neural network functions are only supported for models \
                 declared using the MTK/Catalyst syntax."
            )
        )
    end
    full_ml_calls = PEtab._get_full_ml_calls(prob)
    grouped_calls = PEtab._group_full_ml_calls_by_signature(full_ml_calls)
    nn_idx > length(grouped_calls) && throw(
        ArgumentError(
            "Requested nn_idx=$(nn_idx) but there are only $(length(grouped_calls)) unique \
             fitted functions found."
        )
    )
    nn_sym, ps_sym, all_input_args = grouped_calls[nn_idx]
    length(all_input_args[1]) != 1 && throw(
        ArgumentError(
            "The fitted function plotting recipe only supports fitted functions with a \
             single input. Encountered a function with $(length(all_input_args[1])) inputs."
        )
    )

    # Determines the `x_val`, `y_vals`, and `color` options. These depend on the plot type.
    x_vals, y_vals, _color = if plot_type == :best_function
        _handle_best_function(res, prob, nn_sym, ps_sym, all_input_args, plt_dens, x_support, plotted_dim)
    elseif plot_type == :function_ensemble
        _handle_function_ensemble(
            res, prob, nn_sym, ps_sym, all_input_args, plt_dens,
            x_support, num_plotted_nn, loss_thres, plotted_dim, clustering_function
        )
    else
        throw(ArgumentError("Unsupported fitted neural network plot type: $(plot_type)."))
    end

    # Returns the final plotting info.
    _x_label = String(ModelingToolkitBase.getname(all_input_args[1][1]))
    _y_label = "$(ModelingToolkitBase.getname(nn_sym))($(ModelingToolkitBase.getname(all_input_args[1][1])); $(ModelingToolkitBase.getname(ps_sym)))"
    return _x_label, "", (
            x = x_vals, y = y_vals, color = _color, label = "", seriestype = :line,
            y_label = _y_label,
        )
end

# Function for handling the `plot_type == :best_function` (i.e. plotting the single best fitted function).
# The color defaults to 1
function _handle_best_function(
        res, prob, nn_sym, ps_sym, all_input_args, plt_dens, x_support, plotted_dim
    )
    x_vals, y_vals = _eval_fitted_function(
        res, prob, nn_sym, ps_sym, all_input_args, plt_dens, x_support, plotted_dim
    )
    return x_vals, y_vals, 1
end

# Function for handling the `plot_type == :function_ensemble` (i.e. plotting the ensemble of fitted functions).
# The color defaults to 1
function _handle_function_ensemble(
        res, prob, nn_sym, ps_sym, all_input_args, plt_dens,
        x_support, num_plotted_nn, loss_thres, plotted_dim, clustering_function
    )
    (res isa PEtab.PEtabMultistartResult) || throw(
        ArgumentError(
            "The `:function_ensemble` plot type requires a multi-start calibration run \
             having been performed."
        )
    )

    # Loops through all optimisation runs, and computes the corresponding `x_vals` and `y_vals`.
    # Also find the corresponding color according to the clustering algorithm,
    x_valss = []
    y_valss = []
    color_idxs = []
    for (run_idx, run) in enumerate(res.runs)
        run.fmin > loss_thres && continue
        x_support_i = if isnothing(x_support)
            _get_x_support(nn_sym, all_input_args, run, prob)
        else
            (x_support isa Vector) ? x_support[run_idx] : x_support
        end
        x_vals, y_vals = _eval_fitted_function(
            run, prob, nn_sym, ps_sym, all_input_args, plt_dens, x_support_i, plotted_dim
        )
        push!(x_valss, x_vals)
        push!(y_valss, y_vals)
        push!(color_idxs, run_idx)
    end

    # Checks `num_plotted_nn` and if we should reduce the number of plotted functions.
    if !isnothing(num_plotted_nn)
        if length(x_valss) < num_plotted_nn
            num_plotted_nn = length(x_valss)
            @warn "Requested num_plotted_nn=$(num_plotted_nn) but only $(length(x_valss)) \
                   runs were found with loss below the specified threshold. Plotting all \
                   $(length(x_valss)) runs."
        end
        x_valss = x_valss[1:num_plotted_nn]
        y_valss = y_valss[1:num_plotted_nn]
        color_idxs = color_idxs[1:num_plotted_nn]
    end

    # Determine the color vector using the clustering function.
    colors = clustering_function(res.runs[color_idxs])

    return x_valss, y_valss, colors
end

# For a specific fitted function, and a specific run, returns the fitted function evaluated on a  grid of x-values.
function _eval_fitted_function(
        res::PEtab.EstimationResult, prob::PEtab.PEtabODEProblem, nn_sym,
        ps_sym, all_input_args, plt_dens, x_support, plotted_dim
    )
    # Determines the x-value input that is supported and the x-axis value grid.
    isnothing(x_support) && (x_support = _get_x_support(nn_sym, all_input_args, res, prob))
    x_vals = range(first(x_support), last(x_support); length = plt_dens)

    # Evaluates the fitted function on the input x-grid of values. Sets the default color = 1.
    oprob, _ = PEtab.get_odeproblem(res, prob)
    fitted_nn, fitted_ps = oprob.ps[[nn_sym, ps_sym]]
    f(x) = fitted_nn(x, fitted_ps)
    y_vals = [f(x)[plotted_dim] for x in x_vals]
    return x_vals, y_vals
end

# For a specific fitted function, consider all inputs to it (in all_input_args). This functions
# Simulates the final fitted model, checks all input values to the function occurring in
# these simulations, and determines the x-value support for plotting the function (as the
# minimum/maximum input values occurring within the simulations).
function _get_x_support(nn_sym, all_input_args, res, prob)
    isempty(all_input_args) && throw(
        ArgumentError(
            "Cannot determine x-support for $(nn_sym): no neural-network inputs were found."
        )
    )
    length(first(all_input_args)) != 1 && throw(
        ArgumentError(
            "The function `_get_x_support` currently only supports fitted functions with \
             a single input. Encountered a function with $(length(first(all_input_args))) \
             inputs."
        )
    )

    # Prepare for looping through simulations.
    input_vars = vcat(all_input_args...)
    x_support = Any[nothing, nothing]

    # Loops through all simulations and finds maximum and minimum inputs.
    conditions_ids = prob.model_info.simulation_info.conditionids
    simulation_ids = string.(conditions_ids[:simulation])
    for (i, simulation_id) in pairs(simulation_ids)
        pre_equilibration_id = conditions_ids[:pre_equilibration][i]
        if pre_equilibration_id == :None
            condition = simulation_id
            experiment_id = simulation_id
        else
            condition = Symbol(pre_equilibration_id) => Symbol(simulation_id)
            experiment_id = "$(pre_equilibration_id)=>$(simulation_id)"
        end
        osol = PEtab.get_odesol(res, prob; condition = condition)
        for v in input_vars
            v_min, v_max = extrema(osol[v])
            if x_support[1] === nothing || v_min < x_support[1]
                x_support[1] = v_min
            end
            if x_support[2] === nothing || v_max > x_support[2]
                x_support[2] = v_max
            end
        end
    end

    return Tuple(x_support)
end
