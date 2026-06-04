# [Plotting fitted neural network functions](@id sciml_plotting)
Previously, we described how PEtab implements various plots for evaluating the parameter estimation procedure or the fitted model. Similarly, specialised plotting functionality is provided for investigating fitted neural network functions.

PEtab currently provides two fitted-function plot types:
- Plotting the *single fitted function* discovered from the data.
- Plotting the *ensemble of fitted functions* learned from the data. This can be used as a measure of *functional identifiability*.

!!! note
    Plotting fitted functional forms discovered by neural networks is currently supported only for models created through ModelingToolkitBase or Catalyst.
!!! note
    Plotting fitted functional forms discovered by neural networks is currently supported only for fitted functions with a *single* input argument.
!!! note
    The retrieve the fitted functions as normally usually function (e.g. to investigate the manually), consider using the `PEtab.get_fitted_functions` function.

## Plotting of single fitted functions

For these examples we will use a mutual activation loop model to synthetic data, and use this to demonstrate the plotting functionality. In the model, $X$ and $Y$ activate each other. The effect of $Y$ on $X$'s production follows a Hill function. Here, we will assume that the form of this activation function is unknown, and attempt to recover it using a UDE. In the code below, we declare a model, generate synthetic data, and fit it. The resulting fitting problem (`petab_prob`) and calibration result (`petab_sol`) will be used throughout the remaining tutorial.
```@example 1
# Create model (a mutual activation loop).
using Catalyst
rn = @reaction_network begin
    hill(Y,v,K,n), 0 --> X
    X, 0 --> Y
    d, (X,Y) --> 0
end

# Generate some (synthetic) data for the fitting procedure.
using Distributions, OrdinaryDiffEqTsit5, Plots
t_measurement = 0.0:1:50.0
u0 = [:X => 2.0, :Y => 0.1]
ps_true = [:v => 1.1, :K => 2.0, :n => 3.0, :d => 0.5]
oprob_true = ODEProblem(rn, u0, t_measurement[end], ps_true)
sol_true = solve(oprob_true, Tsit5())
σ = 0.2
X_true = sol_true(t_measurement; idxs = :X)
X_observed = [rand(Normal(X, σ)) for X in X_true]
Y_true = sol_true(t_measurement; idxs = :Y)
Y_observed = [rand(Normal(Y, σ)) for Y in Y_true]
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
    d, (X,Y) --> 0
end

# Create the UDE PEtab problem.
using DataFrames, Optim, PEtab
observables = [PEtabObservable(:obs_X, :X, σ), PEtabObservable(:obs_Y, :Y, σ)]
pest = [PEtabMLParameter(:θ),PEtabParameter(:d; scale = :log10)]
mX = DataFrame(obs_id = "obs_X", time = t_measurement, measurement = X_observed)
mY = DataFrame(obs_id = "obs_Y", time = t_measurement, measurement = Y_observed)
petab_model = PEtabModel(rn_ude, observables, vcat(mX, mY), pest; speciemap = u0)
petab_prob = PEtabODEProblem(petab_model)

# Fit the UDE. Here, we just load a pre-calibrated result (however, we could compute it through `calibrate_multistart`)
path_res = joinpath(@__DIR__, "..", "..", "assets", "optimization_results", "mutual_activation_loop") # hide
petab_sol = PEtabMultistartResult(path_res)
nothing # hide
```

After we have fitted a model containing a neural network, we can use the `plot` function, passing both the fitted model and the original fitting problem, to plot the fitted functional form. To designate that we want to plot the single fitted function (and not some other [type of plot](@ref pest_plotting)), we use the `plot_type = :best_function` option.
```@example 1
plot(petab_sol, petab_prob; plot_type = :best_function)
```
For a multistart optimisation run, this plots the functional form discovered by the neural network in the sub-run with the *best* optimisation loss value. For a single-start run, it simply plots the functional form discovered by the neural network in that single run.

The plot can be modified using standard plotting options.
```@example 1
plot(petab_sol, petab_prob; plot_type = :best_function, lw = 5, color = :purple, linestyle = :dash)
```

By default, PEtab determines the support for the fitted function encountered in the fitted solution, and evaluates the function for these values. That is, in the above problem, the fitted model displays $Y$ values over approximately the range $(0.0, 4.0)$. PEtab determines this and plots the function over this range. To set a custom range, use the `x_support` argument. This can be used, for example, to extrapolate the fitted function beyond where it is supported by the data.
```@example 1
plot(petab_sol, petab_prob; plot_type = :best_function, x_support = (0.0, 5.0))
```

Other available options specific to the `:best_function` plot, with their default values, are:
- `plt_dens = 200`: The fitted function is evaluated on an evenly spaced grid along the designated `x_support` with this number of grid points.
- `nn_idx = 1`: For models where multiple neural networks are fitted, this determines which neural network's functional form to plot.
- `plotted_dim = 1`: Only relevant for functions with multiple output dimensions. This value determines which output dimension to plot.

## Plotting of fitted function ensemble

When performing multistart optimisation, multiple independent fits are computed, each corresponding to a single fitted function. The `plot_type = :function_ensemble` option designates this plot type:
```@example 1
plot(petab_sol, petab_prob; plot_type = :function_ensemble, xlimit = (0.0, 5.0), ylimit = (0.0, 2.0))
```
In practice, these *ensemble plots* can be used to perform *practical functional identifiability analysis*. That is, if the ensemble converges to the same functional form, this suggests that only a single functional form is compatible with the data, and that it is identifiable. By default, PEtab clusters the optimisation runs using the principle described [here](@ref pest_plotting_multirun_clustering). This clustering is then used to determine the colors of the fitted functional forms in the ensemble plot, with forms originating from the same cluster having the same color.

Alternatively, the `loss_thres` option can be used to plot only functional forms with a sufficiently good loss value. That is, while in the above plot we see a wide range of fitted functions, here, we cap it so that plotted functions must achieve a likelihood vaguely in the vicinity of the optimal found likelihood (which might be negative). 
```@example 1
plot(petab_sol, petab_prob; plot_type = :function_ensemble, loss_thres = petab_sol.fmin + abs(petab_sol.fmin)/2)
```
In this plot, all displayed functions are approxiamtely Hill functions (which is also the sought true function)

The `:function_ensemble` plot type also supports the following options:
- `num_plotted_nn`: Specifies how many functions to plot. Defaults to plotting all functions achieving the loss threshold.
- `clustering_function`: The clustering function used to determine colors in the plot. It uses the same syntax as described [here](@ref pest_plotting_multirun_clustering).

The `:function_ensemble` plot type also supports the `plt_dens`, `nn_idx`, `plotted_dim`, and `x_support` options available for `plot_type = :best_function`. By default, PEtab will determine the support for each individual plotted functional form (these are not necessarily the same). If you provide a single tuple argument for `x_support`, it will be used for all fitted functional forms. You can also provide a vector of the same length as the number of plotted functions, providing separate values for each plotted function.
