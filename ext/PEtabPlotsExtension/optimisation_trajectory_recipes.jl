# The possible types of plots avaiable for PEtabOptimisationResult.
const PLOT_TYPES = [:objective, :best_objective]
const PLOT_TYPES_MS = [
    :objective,
    :best_objective,
    :waterfall,
    :runtime_eval,
    :parallel_coordinates
]

# Plots the objective function progression for a PEtabOptimisationResult.
# I wanted to use `yaxis` and not `yaxis_scale`, but that seems prevents the user from
# using `yaxis` to overwrite things (not sure why).
@recipe function f(res::PEtabOptimisationResult; plot_type = :best_objective,
                   yaxis_scale = determine_yaxis([res], plot_type),
                   obj_shift = objective_shift([res], plot_type, yaxis_scale))
    # Checks if any values were recorded.
    if isempty(res.ftrace)
        error("No function evaluations where recorded in the calibration run, was \
               save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, PLOT_TYPES)
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative \
               values are: $(PLOT_TYPES).")
    end

    # Options for different plot types.
    if plot_type == :objective
        # Fixed
        label --> ""
        yaxis --> yaxis_scale
        xlabel --> "Function evaluation"
        yguide --> "Objective value"
        seriestype --> :scatter

        # Derived
        y_vals = res.ftrace .- obj_shift
        x_vals = 1:length(y_vals)

        markershape --> handle_Inf!(y_vals)
    elseif plot_type == :best_objective
        # Fixed
        label --> ""
        yaxis --> yaxis_scale
        xlabel --> "Function evaluation"
        yguide --> "Final nllh value"
        seriestype --> :path

        # Tunable
        lw --> 3

        # Derived
        y_vals = res.ftrace
        for idx in 2:(res.niterations + 1)
            y_vals[idx] = min(res.ftrace[idx], y_vals[idx - 1])
        end
        y_vals = y_vals .- obj_shift
        x_vals = 1:length(y_vals)
        xlimit --> (1, length(x_vals))
    end

    x_vals, y_vals
end

# Plots the objective function progressions for a PEtabMultistartResult.
@recipe function f(res_ms::PEtabMultistartResult;
                   plot_type = :waterfall,
                   best_idxs_n = (plot_type in [:waterfall, :runtime_eval] ?
                                  res_ms.nmultistarts : 10),
                   idxs = best_runs(res_ms, best_idxs_n),
                   clustering_function = objective_value_clustering,
                   yaxis_scale = determine_yaxis(res_ms.runs[idxs], plot_type),
                   obj_shift = objective_shift(res_ms.runs[idxs], plot_type, yaxis_scale))

    # Checks if any values were recorded.
    if plot_type in [:objective, :best_objective] && isempty(res_ms.runs[1].ftrace)
        error("No function evaluations where recorded in the calibration run, was \
               save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, PLOT_TYPES_MS)
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative \
               values are: $(PLOT_TYPES_MS).")
    end

    # Finds the desired type of plot and generates it.
    if plot_type == :objective
        # Fixed
        label --> ""
        yaxis --> yaxis_scale
        xlabel --> "Function evaluation"
        yguide --> "Objective value"
        seriestype --> :scatter

        # Tunable
        ma --> 0.7

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :ftrace)
        y_vals = [yvs .- obj_shift for yvs in y_vals]
        x_vals = [1:l for l in length.(y_vals)]

        markershape --> handle_Inf!(y_vals)
        color --> clustering_function(res_ms.runs[idxs])
    elseif plot_type == :best_objective
        # Fixed
        label --> ""
        yaxis --> yaxis_scale
        xlabel --> "Function evaluation"
        yguide --> "Final nllh value"
        seriestype --> :path

        # Tunable
        lw --> 2
        la --> 0.7

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :ftrace)
        for (run_idx, run) in enumerate(res_ms.runs[idxs])
            for idx in 2:(run.niterations + 1)
                y_vals[run_idx][idx] = min(run.ftrace[idx], y_vals[run_idx][idx - 1])
            end
        end
        y_vals = [yvs .- obj_shift for yvs in y_vals]
        x_vals = [1:l for l in length.(y_vals)]
        color --> clustering_function(res_ms.runs[idxs])
        xlimit --> (1, maximum(length.(x_vals)))
    elseif plot_type == :waterfall
        # Fixed
        label --> ""
        yaxis --> yaxis_scale
        xlabel --> "Optimisation run index"
        yguide --> "Final nllh value"
        seriestype --> :scatter

        # Tunable
        ms --> 8

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :fmin)
        x_vals = sortperm(sortperm(y_vals))
        y_vals = [yvs .- obj_shift for yvs in y_vals]
        markershape --> handle_Inf!(y_vals)
        color --> clustering_function(res_ms.runs[idxs])[1, :]
    elseif plot_type == :runtime_eval
        # Fixed
        label --> ""
        yaxis --> yaxis_scale
        xaxis --> :log10
        xlabel --> "Runtime [s]"
        ylabel --> "Final nllh value"
        seriestype --> :scatter

        # Tunable
        ms --> 6
        ma --> 0.9

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :fmin)
        y_vals = [yvs .- obj_shift for yvs in y_vals]
        x_vals = getfield.(res_ms.runs[idxs], :runtime)
        color --> clustering_function(res_ms.runs[idxs])[1, :]
    elseif plot_type == :parallel_coordinates
        # Fixed
        label --> ""
        xlabel --> "Normalised parameter value"
        ylabel --> "Parameter"
        markershape --> :circle

        # Tunable
        lw --> 2
        la --> 0.7
        ma --> 0.8

        # Derived
        p_mins = [minimum(run.xmin[idx] for run in res_ms.runs[idxs])
                  for idx in 1:length(res_ms.xmin)]
        p_maxs = [maximum(run.xmin[idx] for run in res_ms.runs[idxs])
                  for idx in 1:length(res_ms.xmin)]
        x_vals = [[(p_val - p_min) / (p_max - p_min)
                   for (p_val, p_min, p_max) in zip(run.xmin, p_mins, p_maxs)]
                  for run in res_ms.runs[idxs]]

        y_vals = 1:length(res_ms.xmin)
        yticks --> (y_vals, res_ms.runs[1].xmin |> propertynames)
        color --> clustering_function(res_ms.runs[idxs])
    end

    x_vals, y_vals
end

# Finds the best n runs in among all runs, and return their indexes.
function best_runs(res_ms, n)
    best_idxs = sortperm(getfield.(res_ms.runs, :fmin))
    return best_idxs[(end - min(n, res_ms.nmultistarts) + 1):end]
end

# Converts Infs to the largest non-inf value (and return appropriate markershape vectors).
function handle_Inf!(y_vals::Vector{Float64}; max_val = maximum(filter(!isinf, y_vals)))
    infNan_idxs = isinf.(y_vals) .|| isnan.(y_vals)
    y_vals[infNan_idxs] .= max_val
    return ifelse.(infNan_idxs, :cross, :circle)
end
function handle_Inf!(y_vals::Vector{Vector{Float64}})
    max_val = maximum(filter(!isinf, vcat(y_vals...)))
    reshape(handle_Inf!.(y_vals; max_val = max_val), 1, length(y_vals))
end

# For a multistart optimisation result, clusters the runs according to their bojective value
# (and assign them a number, which corresponds to a colour).
function objective_value_clustering(runs::Vector{PEtabOptimisationResult}; thres = 0.1)
    vals = getfield.(runs, :fmin)
    n = length(vals)
    idxs_v = Pair.(1:n, vals)
    idxs_v_sorted = sort(idxs_v; by = p -> p[2])
    colors = fill(-1, n)
    cur_val = idxs_v_sorted[1][2]
    cur_color = 1
    for (i, v) in idxs_v_sorted
        if v > cur_val + thres
            cur_val = v
            cur_color += 1
        end
        colors[i] = cur_color
    end
    reshape(colors, 1, n)
end

# A helper function which determine whether the y-axis should be logarithmic or not.
# If the range between the highest and lowest plotted value is more than 100, use logarithmic
# y-axis (else, linear).
function determine_yaxis(runs, plot_type::Symbol)
    plot_type ∉ [:objective, :best_objective, :waterfall, :runtime_eval] && return :identity
    min_val, max_val = extrema(get_plotted_vals(runs, plot_type))
    return (max_val/min_val) > 100 ? :log10 : :identity
end

# A helper function which determine whether we need to shift the objective values. Used
# to prevent negative values from being plotted on log-scale. If we shift, all objective
# values are shifted by the same amount, so the lowest one equals to 1.
function objective_shift(runs, plot_type::Symbol, yaxis_scale)
    min_val = minimum(get_plotted_vals(runs, plot_type))
    if (yaxis_scale != :identity) && (min_val <= 0.0)
        return min_val - 1.0
    end
    return 0.0
end

# Returns all loss function values. Which we extract depends on the type of plot.
# This workflow could likely be made more performant.
function get_plotted_vals(runs, plot_type::Symbol)
    vals = getfield.(runs, :fmin)
    if (plot_type == :objective) && !isempty(runs[1].ftrace)
        append!(vals, [run.ftrace[1] for run in runs])
    elseif (plot_type == :best_objective) && !isempty(runs[1].ftrace)
        foreach(run -> append!(vals, run.ftrace), runs)
    end
    return vals
end
