# Tests the plotting recipes for PEtabOptimisationResult and PEtabMultistartResult.
# Written by Torkel Loman.

using DataFrames, Catalyst, OrdinaryDiffEq, Optim, PEtab, Plots, Test

### Preparations ###
petab_ms_res = PEtabMultistartResult(joinpath(@__DIR__, "optimisation_results", "boehm"))

function cumulative_mins(v)
    m = v[1]
    return [m = min(m, x) for x in v]
end
function best_runs(res_ms, n)
    best_idxs = sortperm(getfield.(res_ms.runs, :fmin))
    return best_idxs[end-min(n, res_ms.nmultistarts)+1:end]
end

### Run Tests ###

# Tests objective function evaluations plot.
# Tests idxs functionality.
let
    p_obj = plot(petab_ms_res; plot_type=:objective, idxs=1:5)
    @test p_obj.n == 5
    for i = 1:5
        @test p_obj.series_list[i].plotattributes[:x] == 1:(petab_ms_res.runs[i].niterations+1)
        p_obj.series_list[i].plotattributes[:y] == petab_ms_res.runs[i].ftrace
    end
end

# Tests best objective function evaluations plot.
# Test best_idxs_n functionality.
let
    p_best_obj = plot(petab_ms_res; plot_type=:best_objective, best_idxs_n=15)
    for (i, j) in enumerate(best_runs(petab_ms_res, 15))
        @test p_best_obj.series_list[i].plotattributes[:x] == 1:(petab_ms_res.runs[j].niterations+1)
        @test p_best_obj.series_list[i].plotattributes[:y] == cumulative_mins(petab_ms_res.runs[j].ftrace)
    end
end

# Tests waterfall plot.
let
    p_waterfall = plot(petab_ms_res; plot_type=:waterfall)
    @test p_waterfall.series_list[1].plotattributes[:x] == 1:petab_ms_res.:nmultistarts
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
    xnames = propertynames(petab_ms_res.runs[1].xmin)
    p_parallel_coord.series_list[1].plotattributes[:y] == 1:length(xnames)
end

# Tests the plots comparing the fitted solution to the measurements.
let
    # Declare model
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end

    u0 = [:E => 1.0, :SE => 0.0, :P => 0.0]
    p_true = [:kB => 1.0, :kD => 0.1, :kP => 0.5]

    # Simulate data.
    # Condition 1.
    oprob_true_c1 = ODEProblem(rn,  [:S => 1.0; u0], (0.0, 10.0), p_true)
    true_sol_c1 = solve(oprob_true_c1, Tsit5())
    data_sol_c1 = solve(oprob_true_c1, Tsit5(); saveat=1.0)
    c1_t, c1_E, c1_P = data_sol_c1.t[2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c1[:E][2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c1[:P][2:end]

    # Condition 2.
    oprob_true_c2 = ODEProblem(rn,  [:S => 0.5; u0], (0.0, 10.0), p_true)
    true_sol_c2 = solve(oprob_true_c2, Tsit5())
    data_sol_c2 = solve(oprob_true_c2, Tsit5(); saveat=1.0)
    c2_t, c2_E, c2_P = data_sol_c2.t[2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c2[:E][2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c2[:P][2:end]

    # Make PETab problem.
    @unpack E,P = rn
    obs_E = PEtabObservable(E, 0.5)
    obs_p = PEtabObservable(P, 0.5)
    observables = Dict("obs_E" => obs_E, "obs_p" => obs_p)

    par_kB = PEtabParameter(:kB)
    par_kD = PEtabParameter(:kD)
    par_kP = PEtabParameter(:kP)
    params = [par_kB, par_kD, par_kP]

    c1 = Dict(:S => 1.0)
    c2 = Dict(:S => 0.5)
    simulation_conditions = Dict("c1" => c1, "c2" => c2)

    m_c1_E = DataFrame(simulation_id="c1", obs_id="obs_E", time=c1_t, measurement=c1_E)
    m_c1_P = DataFrame(simulation_id="c1", obs_id="obs_p", time=c1_t, measurement=c1_P)
    m_c2_E = DataFrame(simulation_id="c2", obs_id="obs_E", time=c2_t, measurement=c2_E)
    m_c2_P = DataFrame(simulation_id="c2", obs_id="obs_p", time=c2_t, measurement=c2_P)
    measurements = vcat(m_c1_E, m_c1_P, m_c2_E, m_c2_P)

    # Fit solution
    model = PEtabModel(rn, simulation_conditions , observables, measurements, params;
                       statemap=u0, verbose = false)
    prob = PEtabODEProblem(model; verbose = false)
    res = calibrate_multistart(prob, IPNewton(), 5)

    # Check comparison dictionary.
    comp_dict = get_obs_comparison_plots(res, prob)
    issetequal(keys(comp_dict), ["c1", "c2"])
    issetequal(keys(comp_dict["c1"]), ["obs_E", "obs_p"])
    issetequal(keys(comp_dict["c2"]), ["obs_E", "obs_p"])

    # Make plots
    c1_E_plt = plot(res, prob; obsids=["obs_E"], cid="c1")
    c1_P_plt = plot(res, prob; obsids=["obs_p"], cid="c1")
    c1_E_P_plt = plot(res, prob; cid="c1")

    c2_E_plt = plot(res, prob; obsids=["obs_E"], cid="c2")
    c2_P_plt = plot(res, prob; obsids=["obs_p"], cid="c2")
    c2_E_P_plt = plot(res, prob; cid="c2")

    # Fetch sols.
    sol_c1 = get_odesol(res, prob; cid="c1")
    sol_c2 = get_odesol(res, prob; cid="c2")

    # Test plots.
    for i in 1:2
        @test comp_dict["c1"]["obs_E"].series_list[i].plotattributes[:x] == c1_E_plt.series_list[i].plotattributes[:x]
        @test comp_dict["c1"]["obs_E"].series_list[i].plotattributes[:y] == c1_E_plt.series_list[i].plotattributes[:y]
        @test comp_dict["c1"]["obs_p"].series_list[i].plotattributes[:x] == c1_P_plt.series_list[i].plotattributes[:x]
        @test comp_dict["c1"]["obs_p"].series_list[i].plotattributes[:y] == c1_P_plt.series_list[i].plotattributes[:y]
        @test comp_dict["c2"]["obs_E"].series_list[i].plotattributes[:x] == c2_E_plt.series_list[i].plotattributes[:x]
        @test comp_dict["c2"]["obs_E"].series_list[i].plotattributes[:y] == c2_E_plt.series_list[i].plotattributes[:y]
        @test comp_dict["c2"]["obs_p"].series_list[i].plotattributes[:x] == c2_P_plt.series_list[i].plotattributes[:x]
        @test comp_dict["c2"]["obs_p"].series_list[i].plotattributes[:y] == c2_P_plt.series_list[i].plotattributes[:y]
    end

    @test sol_c1.t == c1_E_plt.series_list[2].plotattributes[:x]
    @test sol_c1.t == c1_P_plt.series_list[2].plotattributes[:x]
    @test sol_c1.t == c1_E_P_plt.series_list[2].plotattributes[:x]
    @test sol_c2.t == c2_E_plt.series_list[2].plotattributes[:x]
    @test sol_c2.t == c2_P_plt.series_list[2].plotattributes[:x]
    @test sol_c2.t == c2_E_P_plt.series_list[2].plotattributes[:x]

    @test sol_c1[:E] == c1_E_plt.series_list[2].plotattributes[:y] == c1_E_P_plt.series_list[2].plotattributes[:y]
    @test sol_c1[:P] == c1_P_plt.series_list[2].plotattributes[:y] == c1_E_P_plt.series_list[4].plotattributes[:y]
    @test sol_c2[:E] == c2_E_plt.series_list[2].plotattributes[:y] == c2_E_P_plt.series_list[2].plotattributes[:y]
    @test sol_c2[:P] == c2_P_plt.series_list[2].plotattributes[:y] == c2_E_P_plt.series_list[4].plotattributes[:y]

    @test c1_E == c1_E_plt.series_list[1].plotattributes[:y] == c1_E_P_plt.series_list[1].plotattributes[:y]
    @test c1_P == c1_P_plt.series_list[1].plotattributes[:y] == c1_E_P_plt.series_list[3].plotattributes[:y]
    @test c2_E == c2_E_plt.series_list[1].plotattributes[:y] == c2_E_P_plt.series_list[1].plotattributes[:y]
    @test c2_P == c2_P_plt.series_list[1].plotattributes[:y] == c2_E_P_plt.series_list[3].plotattributes[:y]
end
