# The possible types of plots avaiable for PEtabOptimisationResult.
const plot_types = [:objective, :best_objective]
const plot_types_ms = [:objective, :best_objective, :waterfall]

# Plots the objective function progression for a PEtabOptimisationResult.
@recipe function f(res::PEtabOptimisationResult{T}; plot_type=:best_objective) where T
    # Checks if any values were recorded.
    if isempty(res.ftrace)
        error("No function evaluations where recorded in the calibration run, was save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, plot_types)
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative values are: $(plot_types).")
    end

    # Sets default values common for all plot types.
    xlabel --> "Function evaluations"
    yguide --> "Best objective value"
    yaxis --> :log10
     
    # Options for different plot types.
    if plot_type == :objective
        seriestype --> :scatter

        x_vals = 1:(res.n_iterations+1)
        y_vals = deepcopy(res.ftrace)

        # In case some values are Infs, reduce these.
        inf_idxs = isinf.(y_vals)
        max_val = maximum(filter(!isinf, y_vals))
        y_vals[inf_idxs] .= max_val
        markershape --> ifelse.(inf_idxs, :cross, :circle)
    elseif plot_type == :best_objective
        seriestype --> :path

        x_vals = 1:(res.n_iterations+1)
        y_vals = deepcopy(res.ftrace)
        for idx in 2:(res.n_iterations+1)
            y_vals[idx] = min(res.ftrace[idx], y_vals[idx-1])
        end
    end

    x_vals, y_vals
end

# Plots the objective function progressions for a PEtabMultistartOptimisationResult.
@recipe function f(res_ms::PEtabMultistartOptimisationResult; plot_type=:best_objective)
    # Checks if any values were recorded.
    if isempty(res_ms.runs[1].ftrace)
        error("No function evaluations where recorded in the calibration run, was save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, plot_types_ms)
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative values are: $(plot_types_ms).")
    end

    # Sets default values common for all plot types.
    yguide --> "Best objective value"
    yaxis --> :log10

    # Options for different plot types.
    if plot_type == :objective
        xlabel --> "Function evaluations"
        seriestype --> :scatter

        y_vals = getfield.(res_ms.runs, :ftrace)
        x_vals = [1:length(best_vals) for best_vals in y_vals]

        # In case some values are Infs, reduce these.
        max_val = maximum(filter(!isinf, vcat(y_vals...)))
        inf_idxs = [isinf.(yvs) for yvs in y_vals]

        markershape --> [ifelse.(iis, :cross, :circle) for iis in inf_idxs]
        for (iis, yvs) in zip(inf_idxs, y_vals)
            yvs[iis] .= max_val
        end
    elseif plot_type == :best_objective
        xlabel --> "Function evaluations"
        seriestype --> :path

        y_vals = getfield.(res_ms.runs, :ftrace)
        for (run_idx,run) in enumerate(res_ms.runs), idx in 2:(run.n_iterations+1)
            y_vals[run_idx][idx] = min(run.ftrace[idx], y_vals[run_idx][idx-1])
        end
        x_vals = [1:length(best_vals) for best_vals in y_vals]
    elseif plot_type == :waterfall
        label --> "" # "Run " .* string.(1:length(res_ms.runs))
        seriestype --> :scatter
        xlabel --> "Optimisation run"
        color --> 1:length(res_ms.runs)

        y_vals = getfield.(res_ms.runs, :fmin)
        x_vals = sortperm(sortperm(y_vals))

        # In case some values are Infs, reduce these.
        inf_idxs = isinf.(y_vals)
        max_val = maximum(filter(!isinf, y_vals))
        y_vals[inf_idxs] .= max_val
        markershape --> ifelse.(inf_idxs, :cross, :circle)
    end
    
    x_vals, y_vals
end
