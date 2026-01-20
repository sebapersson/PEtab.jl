# [Speeding up gradients for large models (adjoint sensitivities)](@id adjoint)

For large models, adjoint sensitivity analysis is typically the most efficient way to
compute gradients [frohlich2017scalable, ma2021comparison](@cite) (mathematical overview is
available in [sapienza2024differentiable](@cite). PEtab.jl supports adjoint methods from
[SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl).

In practice, performance of adjoint sensitivity is driven by three factors: the adjoint
algorithm (quadrature), the Vector–Jacobian product (VJP) backend, and the ODE solver. This
page discusses these options and assumes familiarity with PEtab’s derivative methods (see
[Derivative methods (gradients and Hessians)](@ref gradient_support)). As a working example,
we use the published model Bachmann [bachmann2011division](@cite) model which is available
in the PEtab standard format (see [Importing PEtab problems](@ref import_petab_problem)).
The PEtab files can be downloaded from [here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/bachmann), and given the problem YAML file, the model can be imported
as:

```@example 1
using PEtab
# path_yaml depends on where the problem is stored
path_yaml = joinpath(@__DIR__, "bachmann", "Bachmann_MSB2011.yaml")
path_yaml = joinpath(@__DIR__, "..", "assets", "bachmann", "Bachmann_MSB2011.yaml") # hide
model = PEtabModel(path_yaml)
nothing # hide
```

## Tuning adjoint gradients

The Bachmann model has 25 ODEs and 113 estimated parameters. Even though
`gradient_method = :ForwardDiff` performs best for this model (see below), it is a useful
example for illustrating adjoint tuning. For adjoint gradients, performance is mainly driven
by the following `PEtabODEProblem` options:

1. `odesolver_gradient`: ODE solver and tolerances (`abstol_adj`, `reltol_adj`) used to
   solve the adjoint problem. In many cases `CVODE_BDF()` performs well.
2. `sensealg`: Adjoint algorithm and its Vector–Jacobian product (VJP) backend. PEtab.jl
   supports `InterpolatingAdjoint`, `QuadratureAdjoint`, and `GaussAdjoint` from
   SciMLSensitivity. A key choice for these is the VJP backend; in practice
   `ReverseDiffVJP(true)` is currently the most stable option.

Since `QuadratureAdjoint` is often less robust, this example compares `InterpolatingAdjoint`
and `GaussAdjoint`:

```@example 1
using SciMLSensitivity, Sundials
solver = ODESolver(CVODE_BDF(); abstol_adj = 1e-3, reltol_adj = 1e-6)

petab_prob1 = PEtabODEProblem(
    model;
    gradient_method = :Adjoint,
    odesolver = solver,
    odesolver_gradient = solver,
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
)
petab_prob2 = PEtabODEProblem(
    model;
    gradient_method = :Adjoint,
    odesolver = solver,
    odesolver_gradient = solver,
    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
)
nothing # hide
```

Note that SciMLSensitivity must be loaded to use adjoints. The adjoint ODE-solver tolerances
are set via `abstol_adj`/`reltol_adj` in `ODESolver` and apply only to the adjoint solve.R
Relaxing the adjoint tolerances (relative to the default `1e-8`) can improve
robustness (fewer failed gradient evaluations). Runtime can then be compared as:

```@example 1
using Printf
x = get_x(petab_prob1)
g1, g2 = similar(x), similar(x)
petab_prob1.grad!(g1, x)
petab_prob2.grad!(g2, x)
b1 = @elapsed petab_prob1.grad!(g1, x) # hide
b2 = @elapsed petab_prob2.grad!(g2, x) # hide
@printf("Runtime InterpolatingAdjoint: %.1fs\n", b1)
@printf("Runtime GaussAdjoint: %.1fs\n", b2)
```

In this setup `InterpolatingAdjoint` is faster (this can vary across machines).

Finally, even when `gradient_method = :Adjoint` is fastest, `:ForwardDiff` is preferable if
it is not substantially slower. Adjoint gradients are more likely to fail (they require an
additional difficult backward adjoint solve), and forward-mode methods often produce more
accurate gradients [persson2025petab](@cite).

## References

```@bibliography
Pages = ["adjoint.md"]
Canonical = false
```
