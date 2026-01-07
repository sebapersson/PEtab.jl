# [Speeding up non-stiff models](@id nonstiff_models)

Default `PEtabODEProblem` options are tuned for stiff biological ODE models based pm
[persson2025petab](@cite). For non-stiff models, the default gradient method is often still
a good choice (it mainly depends on the number of estimated parameters), but changing the
ODE solver can substantially reduce runtime.

Explicit solvers are typically fastest for non-stiff models since they avoid solving a
nonlinear system at each solver step. However, often during parameter estimation the
optimizer explores parameter regions where an otherwise non-stiff model becomes stiff, as
observed in benchmarks [persson2025petab](@cite). A robust compromise is therefore to use a
composite solver that automatically switches between non-stiff and stiff methods, for
example:

```julia
petab_prob = PEtabODEProblem(model;
    odesolver = ODESolver(AutoVern7(Rodas5P())),
)
```

For more details on explicit and composite solvers, see the OrdinaryDiffEq.jl solver
[documentation](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).

## References

```@bibliography
Pages = ["nonstiff_models.md"]
Canonical = false
```
