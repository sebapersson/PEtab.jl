# Tests the plotting recipes for PEtabOptimisationResult and PEtabMultistartResult.
# Written by Torkel Loman.

using DataFrames, OrdinaryDiffEqRosenbrock, Catalyst, Optim, PEtab, Plots, Test

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
@testset "Plot objective function" begin
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
    true_sol_c1 = solve(oprob_true_c1, Rodas5P())
    data_sol_c1 = solve(oprob_true_c1, Rodas5P(); saveat=1.0)
    c1_t, c1_E, c1_P = data_sol_c1.t[2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c1[:E][2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c1[:P][2:end]

    # Condition 2.
    oprob_true_c2 = ODEProblem(rn,  [:S => 0.5; u0], (0.0, 10.0), p_true)
    true_sol_c2 = solve(oprob_true_c2, Rodas5P())
    data_sol_c2 = solve(oprob_true_c2, Rodas5P(); saveat=1.0)
    c2_t, c2_E, c2_P = data_sol_c2.t[2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c2[:E][2:end], (0.8 .+ 0.4*rand(10)) .* data_sol_c2[:P][2:end]

    # Make PETab problem.
    @unpack E,P = rn
    observables = [PEtabObservable("obs_E", E, 0.5),
                   PEtabObservable("obs_p", P, 0.5)]

    par_kB = PEtabParameter(:kB)
    par_kD = PEtabParameter(:kD)
    par_kP = PEtabParameter(:kP)
    params = [par_kB, par_kD, par_kP]

    simulation_conditions = [PEtabCondition(:c1, :S => 1.0),
                             PEtabCondition(:c2, :S => 0.5)]

    m_c1_E = DataFrame(simulation_id="c1", obs_id="obs_E", time=c1_t, measurement=c1_E)
    m_c1_P = DataFrame(simulation_id="c1", obs_id="obs_p", time=c1_t, measurement=c1_P)
    m_c2_E = DataFrame(simulation_id="c2", obs_id="obs_E", time=c2_t, measurement=c2_E)
    m_c2_P = DataFrame(simulation_id="c2", obs_id="obs_p", time=c2_t, measurement=c2_P)
    measurements = vcat(m_c1_E, m_c1_P, m_c2_E, m_c2_P)

    # Fit solution
    model = PEtabModel(rn , observables, measurements, params; speciemap=u0,
                       simulation_conditions = simulation_conditions)
    prob = PEtabODEProblem(model; verbose = false)
    res = calibrate_multistart(prob, IPNewton(), 20)

    # Check comparison dictionary.
    comp_dict = get_obs_comparison_plots(res, prob)
    issetequal(keys(comp_dict), ["c1", "c2"])
    issetequal(keys(comp_dict["c1"]), ["obs_E", "obs_p"])
    issetequal(keys(comp_dict["c2"]), ["obs_E", "obs_p"])

    # Make plots
    c1_E_plt = plot(res, prob; obsids=["obs_E"], cid="c1")
    c1_P_plt = plot(res, prob; obsids=["obs_p"], cid="c1")
    c1_E_P_plt = plot(res, prob; cid="c1")

    c2_E_plt = plot(res.xmin, prob; obsids=["obs_E"], cid="c2", obsid_label = true)
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

    @test sol_c1[:E] == c1_E_plt.series_list[2].plotattributes[:y]
    @test sol_c1[:P] == c1_P_plt.series_list[2].plotattributes[:y]
    @test sol_c2[:E] == c2_E_plt.series_list[2].plotattributes[:y]
    @test sol_c2[:P] == c2_P_plt.series_list[2].plotattributes[:y]

    @test c1_E == c1_E_plt.series_list[1].plotattributes[:y]
    @test c1_P == c1_P_plt.series_list[1].plotattributes[:y]
    @test c2_E == c2_E_plt.series_list[1].plotattributes[:y]
    @test c2_P == c2_P_plt.series_list[1].plotattributes[:y]
end

# Check model fit plotting works for models with pre-eq simulations
let
    rn = @reaction_network begin
        @parameters S0 c3=1.0
        @species S(t)=S0
        c1, S + E --> SE
        c2, SE --> S + E
        c3, SE --> P + E
    end
    speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]

    @unpack E, S, P = rn
    @parameters sigma
    observables = [PEtabObservable("obs_p", P, sigma),
                   PEtabObservable("obs_sum", S + E, 3.0)]
    p_c1 = PEtabParameter(:c1)
    p_c2 = PEtabParameter(:c2)
    p_sigma = PEtabParameter(:sigma)
    pest = [p_c1, p_c2, p_sigma]
    conds = [PEtabCondition(:cond1, :S0 => 3.0),
             PEtabCondition(:cond2, :S0 => 5.0),
             PEtabCondition(:cond_preeq, :S0 => 2.0)]
    measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                            pre_eq_id=["cond_preeq", "cond_preeq", "cond_preeq", "cond_preeq"],
                            obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                            time=[1.0, 10.0, 1.0, 20.0],
                            measurement=[2.5, 50.0, 2.6, 51.0])

    model = PEtabModel(rn, observables, measurements, pest;
                    simulation_conditions = conds, speciemap = speciemap)
    petab_prob = PEtabODEProblem(model)
    x = [0.2070820996670734, 2.6802649314502975, -1.0764046246919647]
    sol = get_odesol(x, petab_prob; cid = :cond1, preeq_id = :cond_preeq)
    p = plot(x, petab_prob; linewidth = 2.0, cid = :cond1)
    @test all(sol[:P] .== p.series_list[2].plotattributes[:y])
    p = plot(x, petab_prob; linewidth = 2.0, cid = :cond1, preeq_id = :cond_preeq)
    @test all(sol[:P] .== p.series_list[2].plotattributes[:y])
    @test_throws AssertionError begin
        p = plot(x, petab_prob; linewidth = 2.0, cid = :cond1, preeq_id = :cond2)
    end
    plots = get_obs_comparison_plots(x, petab_prob)
    @test all(collect(keys(plots)) .== ["pre_cond_preeq_main_cond2", "pre_cond_preeq_main_cond1"])
end
