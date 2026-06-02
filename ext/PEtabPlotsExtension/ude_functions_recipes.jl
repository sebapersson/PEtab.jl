function _plot_ude_function(
		res::PEtab.EstimationResult, prob::PEtabODEProblem;
		nn_idx = 1, plt_dens = 200, x_support = nothing, plotted_dim = 1
	)

    # Extract all the ML functional calls within the model.
    # Check that `nn_idx` is OK. Checks that the model have one input and one input.
    full_ml_calls = _get_full_ml_calls(prob)
    grouped_calls = _group_full_ml_calls_by_signature(full_ml_calls)
    nn_idx > length(grouped_calls) && throw(ArgumentError(
        "Requested nn_idx=$(nn_idx) but there are only $(length(grouped_calls)) unique fitted functions found."
    ))
    nn_sym, ps_sym, all_input_args = grouped_calls[nn_idx]
    length(all_input_args[1]) != 1 && throw(ArgumentError(
        "The fitted function plotting recipe only supports fitted functions with a single input. Encountered a function with $(length(all_input_args[1])) inputs."
    ))

    # Determines the x-value input that is supported and the x-axis value grid.
    isnothing(x_support) && (x_support = _get_x_support(grouped_calls[nn_idx], res, prob))
    x_vals = range(first(x_support), last(x_support); length = plt_dens)

    # Evaluates the fitted function on the input x-grid of values.
    oprob, _ = get_odeproblem(res, prob)
    fitted_nn, fitted_ps = oprob.ps[[nn_sym, ps_sym]]
    f(x) = fitted_nn(x, fitted_ps)
    y_vals = [f(x)[plotted_dim] for x in x_vals]

    # Returns the final plotting structure.
    _color = 1
    _x_label = String(ModelingToolkitBase.getname(all_input_args[1][1]))
    _y_label = "$(ModelingToolkitBase.getname(nn_sym))($(ModelingToolkitBase.getname(all_input_args[1][1])); $(ModelingToolkitBase.getname(ps_sym))))"
    _seriestype = :line
    return (
        x = x_vals, y = y_vals, color = _color, label = "",
        seriestype = _seriestype, y_label = _y_label,
    )
end

function _plot_ude_functions(
		res::PEtab.EstimationResult, prob::PEtabODEProblem;
		nn_idx = 1, plt_dens = 200, x_support = nothing, num_plotted_nn = nothing, loss_thres = Inf, plotted_dim = 1
	)
    # Not Implemented.
end

# For a specific fitted function, consider all inputs to it (in all_input_args). This functions
# Simulates the final fitted model, checks all input values to the function occurring in
# these simulations, and determines the x-value support for plotting the function (as the
# minimum/maximum input values occurring within the simulations).
function _get_x_support(grouped_call, res, prob)
    nn_sym, _, all_input_args = grouped_call
    isempty(all_input_args) && throw(ArgumentError(
        "Cannot determine x-support for $(nn_sym): no neural-network inputs were found."
    ))

    n_inputs = length(first(all_input_args))
    any(length(input_args) != n_inputs for input_args in all_input_args) && throw(
        ArgumentError(
            "Cannot determine x-support for $(nn_sym): inconsistent input arity across ML calls."
        )
    )

    mins = fill(Inf, n_inputs)
    maxs = fill(-Inf, n_inputs)

    for experiment_id in prob.model_info.simulation_info.conditionids[:experiment]
        sys, _, p, _ = PEtab.get_system(res, prob; experiment = experiment_id)
        sol = PEtab.get_odesol(res, prob; experiment = experiment_id)
        state_syms = collect(ModelingToolkitBase.unknowns(sys))
        state_expr = isempty(state_syms) ? nothing : Symbolics.unwrap(first(state_syms))
        iv = (!isnothing(state_expr) && Symbolics.iscall(state_expr)) ?
            first(Symbolics.arguments(state_expr)) : nothing

        base_subs = Dict{Any, Any}(Dict(p))
        for (time_idx, t) in pairs(sol.t)
            subs = copy(base_subs)
            for (state_sym, state_val) in zip(state_syms, sol[:, time_idx])
                subs[state_sym] = state_val
            end
            !isnothing(iv) && (subs[iv] = t)

            for input_args in all_input_args, (arg_idx, input_arg) in pairs(input_args)
                value = Symbolics.substitute(Symbolics.unwrap(input_arg), subs) |> Symbolics.unwrap
                value isa Number || (value = Symbolics.value(value))
                if value isa Number && isfinite(value)
                    value = float(value)
                    mins[arg_idx] = min(mins[arg_idx], value)
                    maxs[arg_idx] = max(maxs[arg_idx], value)
                end
            end
        end
    end

    any(isfinite, mins) || throw(ArgumentError(
        "Cannot determine x-support for $(nn_sym): all evaluated inputs were non-finite."
    ))

    n_inputs == 1 && return range(mins[1], maxs[1]; length = 101)
    return [range(min_val, max_val; length = 101) for (min_val, max_val) in zip(mins, maxs)]
end
