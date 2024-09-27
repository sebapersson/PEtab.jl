# [Options for Non-Biology (Non-Stiff) Models](@id nonstiff_models)

The default options when creating a `PEtabODEProblem` in PEtab.jl are based on extensive benchmarks for dynamic models in biology. A key feature of ODEs in biology is that they are often stiff [stadter2021benchmarking](@cite). While an exact definition of stiffness is elusive, informally, explicit (non-stiff) solvers struggle to efficiently solve stiff models. Whether a model is stiff does not impact the choice of the optimal gradient method, as this depends on the number of parameters to estimate rather than the ODE solver. However, for non-stiff models, using a different ODE solver than the default in PEtab.jl can drastically reduce runtime.

If a problem is non-stiff, it is much more computationally efficient to use an explicit solver, as a non-linear system does not need to be solved at each iteration. However, choosing a purely explicit (non-stiff) solver is often not ideal. When performing multi-start parameter estimation using random initial points, benchmarks have shown that even if the model is non-stiff around the best parameter values, this may not hold for random parameter values. Therefore, for non-stiff (or mildly stiff) models, a good compromise is to use composite solvers that can automatically switch between stiff and non-stiff solvers. Therefore, a good setup often is:

```julia
petab_prob = PEtabODEProblem(model; odesolver=ODESolver(AutoVern7(Rodas5P())))
```

This ODE solver automatically switches between solvers based on stiffness. For more details on non-stiff solver choices, as well as composite solvers, see the [documentation](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) for OrdinaryDiffEq.jl.
