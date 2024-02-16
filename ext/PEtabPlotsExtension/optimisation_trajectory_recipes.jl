# The possible types of plots avaiable for PEtabOptimisationResult.
const plot_types = [:objective, :best_objective]
const plot_types_ms = [
    :objective,
    :best_objective,
    :waterfall,
    :runtime_eval,
    :parallel_coordinates
]

# Plots the objective function progression for a PEtabOptimisationResult.
@recipe function f(res::PEtabOptimisationResult{T}; plot_type = :best_objective) where {T}
    # Checks if any values were recorded.
    if isempty(res.ftrace)
        error("No function evaluations where recorded in the calibration run, was save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, plot_types)
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative values are: $(plot_types).")
    end

    # Options for different plot types.
    if plot_type == :objective
        # Fixed
        label --> ""
        yaxis --> :log10
        xlabel --> "Function evaluation"
        yguide --> "Objective value"
        seriestype --> :scatter

        # Derived
        y_vals = res.ftrace
        x_vals = 1:length(y_vals)

        markershape --> handle_Inf!(y_vals)
    elseif plot_type == :best_objective
        # Fixed
        label --> ""
        yaxis --> :log10
        xlabel --> "Function evaluation"
        yguide --> "Best objective value"
        seriestype --> :path

        # Tunable
        lw --> 3

        # Derived
        y_vals = res.ftrace
        for idx in 2:(res.n_iterations + 1)
            y_vals[idx] = min(res.ftrace[idx], y_vals[idx - 1])
        end
        x_vals = 1:length(y_vals)
        xlimit --> (1, length(x_vals))
    end

    x_vals, y_vals
end

# Plots the objective function progressions for a PEtabMultistartOptimisationResult.
@recipe function f(res_ms::PEtabMultistartOptimisationResult;
                   plot_type = :waterfall,
                   best_idxs_n = (plot_type in [:waterfall, :runtime_eval] ?
                                  res_ms.n_multistarts : 10),
                   idxs = best_runs(res_ms, best_idxs_n),
                   clustering_function = objective_value_clustering)

    # Checks if any values were recorded.
    if isempty(res_ms.runs[1].ftrace)
        error("No function evaluations where recorded in the calibration run, was save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, plot_types_ms)
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative values are: $(plot_types_ms).")
    end

    # Finds the desired type of plot and generates it.
    if plot_type == :objective
        # Fixed
        label --> ""
        yaxis --> :log10
        xlabel --> "Function evaluation"
        yguide --> "Objective value"
        seriestype --> :scatter

        # Tunable
        ma --> 0.7

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :ftrace)
        x_vals = [1:l for l in length.(y_vals)]

        markershape --> handle_Inf!(y_vals)
        color --> clustering_function(res_ms.runs[idxs])
    elseif plot_type == :best_objective
        # Fixed
        label --> ""
        yaxis --> :log10
        xlabel --> "Function evaluation"
        yguide --> "Best objective value"
        seriestype --> :path

        # Tunable
        lw --> 2
        la --> 0.7

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :ftrace)
        for (run_idx, run) in enumerate(res_ms.runs[idxs])
            for idx in 2:(run.n_iterations + 1)
                y_vals[run_idx][idx] = min(run.ftrace[idx], y_vals[run_idx][idx - 1])
            end
        end
        x_vals = [1:l for l in length.(y_vals)]
        color --> clustering_function(res_ms.runs[idxs])
        xlimit --> (1, maximum(length.(x_vals)))
    elseif plot_type == :waterfall
        # Fixed
        label --> ""
        yaxis --> :log10
        xlabel --> "Optimisation run index"
        yguide --> "Best objective value"
        seriestype --> :scatter

        # Tunable
        ms --> 8

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :fmin)
        x_vals = sortperm(sortperm(y_vals))
        markershape --> handle_Inf!(y_vals)
        color --> clustering_function(res_ms.runs[idxs])[1, :]
    elseif plot_type == :runtime_eval
        # Fixed
        label --> ""
        yaxis --> :log10
        xlabel --> "Runtime (s)"
        ylabel --> "Best objective value"
        seriestype --> :scatter

        # Tunable
        ms --> 6
        ma --> 0.9

        # Derived
        y_vals = getfield.(res_ms.runs[idxs], :fmin)
        x_vals = getfield.(res_ms.runs[idxs], :runtime)
        color --> clustering_function(res_ms.runs[idxs])[1, :]
    elseif plot_type == :parallel_coordinates
        # Fixed
        label --> ""
        xlabel --> "Parameter"
        ylabel --> "Normalised parameter value"
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
        yticks --> (y_vals, res_ms.xnames)
        color --> clustering_function(res_ms.runs[idxs])
    end

    x_vals, y_vals
end

# Finds the best n runs in among all runs, and return their indexes.
function best_runs(res_ms, n)
    best_idxs = sortperm(getfield.(res_ms.runs, :fmin))
    return best_idxs[(end - min(n, res_ms.n_multistarts) + 1):end]
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

# For a multistart optimisation result, clusters the runs according to their bojective value (and assign them a number, which corresponds to a colour).
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
