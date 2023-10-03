# Tests the plotting recipes for PEtabOptimisationResult and PEtabMultistartOptimisationResult.
# Written by Torkel Loman.
# Comment: I am only aware of how to load PEtabMultistartOptimisationResult from folders (and not PEtabOptimisationResult), hence I am only testing on teh former (as I don't want to actually run an optimiser within these tests).


# Fetch packages.
using PEtab
using Plots
using Tests

### Preparations ###
petab_ms_res = PEtabMultistartOptimisationResult(joinpath(@__DIR__, "Optimisation_results", "boehm"))

# Helper functions.
function cumulative_mins(v)
    m = v[1]
    return [m = min(m, x) for x in v]
end
function best_runs(res_ms, n)
    best_idxs = sortperm(getfield.(res_ms.runs, :fmin))
    return best_idxs[end-min(n, res_ms.n_multistarts)+1:end]
end

### Run Tests ###

# Tests objective function evaluations plot.
# Tests idxs functionality.
let
    p_obj = plot(petab_ms_res; plot_type=:objective, idxs=1:5)
    @test p_obj.n == 5
    for i = 1:5
        @test p_obj.series_list[i].plotattributes[:x] == 1:(petab_ms_res.runs[i].n_iterations+1)
        p_obj.series_list[i].plotattributes[:y] == petab_ms_res.runs[i].ftrace
    end
end

# Tests best objective function evaluations plot.
# Test best_idxs_n functionality.
let
    p_best_obj = plot(petab_ms_res; plot_type=:best_objective, best_idxs_n=15)
    for (i, j) in enumerate(best_runs(petab_ms_res, 15))
        @test p_best_obj.series_list[i].plotattributes[:x] == 1:(petab_ms_res.runs[j].n_iterations+1)
        @test p_best_obj.series_list[i].plotattributes[:y] == cumulative_mins(petab_ms_res.runs[j].ftrace)
    end
end

# Tests waterfall plot.
let
    p_waterfall = plot(petab_ms_res; plot_type=:waterfall)
    @test p_waterfall.series_list[1].plotattributes[:x] == 1:petab_ms_res.:n_multistarts
    @test p_waterfall.series_list[1].plotattributes[:y] == sort(getfield.(petab_ms_res.runs, :fmin))
end

# Tests runtime evaluation plot.
# Tests idxs functionality. 
let
    p_run_eval = plot(petab_ms_res; plot_type=:runtime_eval, idxs=1:25)
    @test p_run_eval.series_list[1].plotattributes[:x] == getfield.(petab_ms_res.runs[1:25], :runtime)
    @test p_run_eval.series_list[1].plotattributes[:y] == getfield.(petab_ms_res.runs[1:25], :fmin)
end

# Tests parallel coordinates plot.
let
    p_parallel_coord = plot(petab_ms_res; plot_type=:parallel_coordinates, idxs=1:25)
    
    p_mins = [minimum(run.xmin[idx] for run in petab_ms_res.runs[1:25]) for idx in 1:length(petab_ms_res.xmin)]
    p_maxs = [maximum(run.xmin[idx] for run in petab_ms_res.runs[1:25]) for idx in 1:length(petab_ms_res.xmin)]
    x_vals = [[(p_val-p_min)/(p_max-p_min) for (p_val,p_min,p_max) in zip(run.xmin,p_mins,p_maxs)] for run in petab_ms_res.runs[1:25]]

    for idx = 1:25
        @test p_parallel_coord.series_list[idx].plotattributes[:x] == x_vals[idx]
    end
    p_parallel_coord.series_list[1].plotattributes[:y] == 1:length(petab_ms_res.xnames)
end


