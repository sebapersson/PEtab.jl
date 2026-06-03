# generates a fit to carry the tests out on.
# Long term, when UDE fits can be saved, we could move to a saved fit to reduce runtime by a bit (and also try on more types of fits).
begin
    # Sets an rng.
    using StableRNGs
    rng = StableRNG(1)

    # Create model (an extended self-activation loop).
    using Catalyst
    rn = @reaction_network begin
        hill(Y, v, K, n), 0 --> X
        X, 0 --> Y
        d, (X, Y) --> 0
    end

    # Generate some (synthetic) data for the fitting procedure.
    using Distributions, OrdinaryDiffEqDefault, Plots
    t_measurement = 0.0:2.5:50.0
    u0 = [:X => 2.0, :Y => 0.1]
    ps_true = [:v => 1.1, :K => 2.0, :n => 3.0, :d => 0.5]
    oprob_true = ODEProblem(rn, u0, t_measurement[end], ps_true)
    sol_true = solve(oprob_true)
    σ = 0.02
    X_true = sol_true(t_measurement; idxs = :X)
    X_observed = [rand(rng, Normal(X, σ * X)) for X in X_true]
    Y_true = sol_true(t_measurement; idxs = :Y)
    Y_observed = [rand(rng, Normal(Y, σ * Y)) for Y in Y_true]
    plot(sol_true; label = ["X (true)" "Y (true)"], color = [1 2])
    plot!(t_measurement, X_observed; label = "X (measured)", color = 1, seriestype = :scatter)
    plot!(t_measurement, Y_observed; label = "Y (measured)", color = 2, seriestype = :scatter)

    # Create the UDE.
    using ModelingToolkitNeuralNets, Lux
    nn_arch = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )
    @SymbolicNeuralNetwork U, θ = nn_arch
    A(x) = U(x, θ)[1]
    rn_ude = @reaction_network begin
        $A(Y), 0 --> X
        X, 0 --> Y
        d, (X, Y) --> 0
    end

    # Create the UDE PEtabproblem.
    using DataFrames, Optim, PEtab
    observables = [PEtabObservable(:obs_X, :X, σ), PEtabObservable(:obs_Y, :Y, σ)]
    pest = [PEtabMLParameter(:θ), PEtabParameter(:d; scale = :log10)]
    mX = DataFrame(obs_id = "obs_X", time = t_measurement, measurement = X_observed)
    mY = DataFrame(obs_id = "obs_Y", time = t_measurement, measurement = Y_observed)
    petab_model = PEtabModel(rn_ude, observables, vcat(mX, mY), pest; speciemap = u0)
    petab_prob = PEtabODEProblem(petab_model)

    # Fit the UDE
    using Optimisers
    @time petab_sol = calibrate_multistart(
        rng, petab_prob, Optimisers.Adam(1.0e-3), 5;
        options = OptimisersOptions(iterations = 10000)
    )
end

### Test basic Plot Functionality ###

# Checks basic content of the `best_function` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :best_function)
@test length(p_obj.series_list) == 1 # Only a single line is plotted.
@test length(p_obj.series_list[1].plotattributes[:x]) == 200 # Default plot density is 200.
@test p_obj.series_list[1].plotattributes[:x][1] ≈ 0.1 atol = 1.0e-1 rtol = 1.0e-1 # X supports starts (very roughly) at 0.1.
@test p_obj.series_list[1].plotattributes[:x][end] ≈ 4.0 atol = 1.0e-1 rtol = 1.0e-1 # X supports end (very roughly) at 4.0.
true_func(y) = 1.1 * (y^3) / (2^3 + y^3)
@test p_obj.series_list[1].plotattributes[:y][1] ≈ true_func(p_obj.series_list[1].plotattributes[:x][1]) atol = 1.0e-1 rtol = 1.0e-1 # Function approximately correct for low x input.
@test p_obj.series_list[1].plotattributes[:y][100] ≈ true_func(p_obj.series_list[1].plotattributes[:x][100]) atol = 1.0e-1 rtol = 1.0e-1 # Function approximately correct for medium x input.
@test p_obj.series_list[1].plotattributes[:y][end] ≈ true_func(p_obj.series_list[1].plotattributes[:x][end]) atol = 1.0e-1 rtol = 1.0e-1 # Function approximately correct for high x input.

# Checks the `x_support` option of the `best_function` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :best_function, x_support = (1.0, 2.0))
@test p_obj.series_list[1].plotattributes[:x][1] == 1.0 # X support start is correct.
@test p_obj.series_list[1].plotattributes[:x][end] == 2.0 # X support end is correct.

# Checks the `plt_dens` option of the `best_function` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :best_function, plt_dens = 100)
@test length(p_obj.series_list[1].plotattributes[:x]) == 100 # Default plot density is 100.

# Checks basic content of the `function_ensemble` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :function_ensemble)
@test length(p_obj.series_list) == length(petab_sol.runs) # One line is plotted for each run.
for i in 1:length(petab_sol.runs) # Checks that the individual ensemble plots correspond to the best for specific runs.
    p_obj_single = plot(petab_sol.runs[i], petab_prob; plot_type = :best_function)
    @test p_obj.series_list[i].plotattributes[:x] == p_obj_single.series_list[1].plotattributes[:x]
    @test p_obj.series_list[i].plotattributes[:y] == p_obj_single.series_list[1].plotattributes[:y]
end

# Checks the `x_support` option of the `function_ensemble` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :function_ensemble, x_support = (1.0, 6.0))
for i in 1:length(petab_sol.runs)
    @test p_obj.series_list[i].plotattributes[:x][1] == 1.0 # X support start is correct.
    @test p_obj.series_list[i].plotattributes[:x][end] == 6.0 # X support end is correct.
end
p_obj = plot(petab_sol, petab_prob; plot_type = :function_ensemble, x_support = [(1.0, 1.0 + i) for i in 1:length(petab_sol.runs)])
for i in 1:length(petab_sol.runs)
    @test p_obj.series_list[i].plotattributes[:x][1] == 1.0 # X support start is correct.
    @test p_obj.series_list[i].plotattributes[:x][end] == 1.0 + i # X support end is correct.
end

# Checks the `plt_dens` option of the `function_ensemble` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :function_ensemble, plt_dens = 300)
for i in 1:length(petab_sol.runs)
    @test length(p_obj.series_list[i].plotattributes[:x]) == 300 # Plot density is correct.
end

# Checks the `num_plotted_nn` option of the `function_ensemble` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :function_ensemble, num_plotted_nn = 3)
@test length(p_obj.series_list) == 3

# Checks the `loss_thres` option of the `function_ensemble` plot.
p_obj = plot(petab_sol, petab_prob; plot_type = :function_ensemble, loss_thres = petab_sol.fmin + eps())
@test length(p_obj.series_list) == 1 # Only a single function should achieve the minimum loss.
