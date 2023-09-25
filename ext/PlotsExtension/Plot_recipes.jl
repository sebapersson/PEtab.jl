# The possible types of plots avaiable for PEtabOptimisationResult.
const plot_types = [:objective, :best_objective]

# Plots the objective function progression for a PEtabOptimisationResult.
@recipe function f(res::PEtabOptimisationResult{T}; plot_type=:best_objective) where T
    # Checks if any values were recorded.
    if isempty(res.ftrace)
        error("No function evaluations where recorded in the calibration run, was save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, [:objective, :best_objective])
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative values are: $(plot_types).")
    end

    # Sets default values.
    xlabel --> "Function evaluations"
    yguide --> "Best objective value"
    yaxis --> :log10
    seriestype --> (plot_type==:objective ? :scatter : :path)
     
    # Finds the objective function values to plot.
    best_vals = deepcopy(res.ftrace)
    if plot_type == :best_objective
        for idx in 2:(res.n_iterations+1)
            best_vals[idx] = min(res.ftrace[idx], best_vals[idx-1])
        end
    end
    1:(res.n_iterations+1), best_vals
end

# Plots the objective function progressions for a PEtabMultistartOptimisationResult.
@recipe function f(res_ms::PEtabMultistartOptimisationResult; plot_type=:best_objective)
    # Checks if any values were recorded.
    if isempty(res_ms.runs[1].ftrace)
        error("No function evaluations where recorded in the calibration run, was save_trace=true?")
    end

    # Checks that a recognised plot type was used.
    if !in(plot_type, [:objective, :best_objective])
        error("Argument plot_type have an unrecognised value ($(plot_type)). Alternative values are: $(plot_types).")
    end

    # Sets default values.
    xlabel --> "Function evaluations"
    yguide --> "Best objective value"
    yaxis --> :log10
    seriestype --> (plot_type==:objective ? :scatter : :path)
     
    # Finds the objective function values to plot.
    best_valss_ms = getfield.(res_ms.runs, :ftrace)
    if plot_type == :best_objective
        for (run_idx,run) in enumerate(res_ms.runs), idx in 2:(run.n_iterations+1)
            best_valss_ms[run_idx][idx] = min(run.ftrace[idx], best_valss_ms[run_idx][idx-1])
        end
    end
    [1:length(best_vals) for best_vals in best_valss_ms], best_valss_ms
end
