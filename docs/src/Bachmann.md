# [Adjoint Sensitivity Analysis (large models)](@id adjoint)

Having access to the gradient is beneficial for parameter estimation, as gradient-based optimization algorithms often perform best [raue2013lessons, raue2013lessons](@cite). For large model, the most efficient gradient method is adjoint sensitivity analysis [frohlich2017scalable, ma2021comparison](@cite), with a good mathematical description provided in [sapienza2024differentiable](@cite). PEtab.jl supports the adjoint sensitivity algorithms in [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl). For these algorithms, three key options impact performance: which algorithm is used to compute the gradient quadrature, which method is used to compute the Vector-Jacobian-Product (VJP) in the adjoint ODE, and which ODE solver is used. This advanced example covers these considerations and assumes familiarity with gradient methods in PEtab (see [this](@ref gradient_support) page). In addition to this page, further details on tunable options are available in the SciMLSensitivity [documentation](https://github.com/SciML/SciMLSensitivity.jl).

As a working example, we use a published signaling model referred to as the Bachhman model after the first author [bachmann2011division](@cite). The Bachmann model is available in the PEtab standard format (a tutorial on importing problems in the standard format can be found [here](@ref import_petab_problem)), and the PEtab files for this model can be downloaded from [here](https://github.com/sebapersson/PEtab.jl/docs/src/assets/bachmann). Given the problem YAML file, we can import the problem as:

```@example 1
using PEtab
path_yaml = joinpath(@__DIR__, "assets", "bachmann", "Bachmann_MSB2011.yaml")
model = PEtabModel(path_yaml)
nothing # hide
```

## Tuning Options

The Bachmann model is a medium-sized model with 25 species in the ODE system and 113 parameters to estimate. Even though `gradient_method=:ForwardDiff` performs best for this model (more on this below), it is a good example for showcasing different tuning options. In particular, when computing the gradient via adjoint sensitivity analysis, the key tunable options for a `PEtabODEProblem` are:

1. `odesolver_gradient`: Which ODE solver and solver tolerances (`abstol` and `reltol`) to use when solving the adjoint ODE system. Currently, `CVODE_BDF()` performs best.
2. `sensealg`: Which adjoint algorithm to use. PEtab.jl supports the `InterpolatingAdjoint`, `QuadratureAdjoint`, and `GaussAdjoint` methods from SciMLSensitivity. For these, the most important tunable option is the VJP method, where `EnzymeVJP` often performs best. If this method does not work, `ReverseDiffVJP(true)` is a good alternative.

As `QuadratureAdjoint` is the least reliable method, we here explore `InterpolatingAdjoint` and `GaussAdjoint`:

```@example 1
using SciMLSensitivity, Sundials
osolver = ODESolver(CVODE_BDF(); abstol_adj = 1e-3, reltol_adj = 1e-6)
petab_prob1 = PEtabODEProblem(model; gradient_method = :Adjoint,
                              odesolver = osolver, odesolver_gradient = osolver,
                              sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()))
petab_prob2 = PEtabODEProblem(model; gradient_method = :Adjoint,
                              odesolver = osolver, odesolver_gradient = osolver,
                              sensealg = GaussAdjoint(autojacvec = EnzymeVJP()))
nothing # hide
```

Two things should be noted here. First, to use the adjoint functionality in PEtab.jl, SciMLSensitivity must be loaded. Second, when creating the `ODESolver`, `adj_abstol` sets the tolerances for solving the adjoint ODE (but not the standard forward ODE). From our experience, setting the adjoint tolerances lower than the default `1e-8` improves simulation stability (gradient computations fail less frequently). Given this, we can now compare runtime:

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

In this case `InterpolatingAdjoint` performs best (this can change dependent on computer). As mentioned above, another important argument is the VJP method; let us explore the best two options for `InterpolatingAdjoint`:

```@example 1
petab_prob1 = PEtabODEProblem(model; gradient_method = :Adjoint,
                              odesolver = osolver, odesolver_gradient = osolver,
                              sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()))
petab_prob2 = PEtabODEProblem(model; gradient_method = :Adjoint,
                              odesolver = osolver, odesolver_gradient = osolver,
                              sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
petab_prob1.grad!(g1, x) # hide
petab_prob2.grad!(g2, x) # hide
b1 = @elapsed petab_prob1.grad!(g1, x)
b2 = @elapsed petab_prob2.grad!(g2, x)
@printf("Runtime EnzymeVJP() : %.1fs\n", b1)
@printf("Runtime ReverseDiffVJP(true): %.1fs\n", b2)                              
nothing # hide
```

In this case, `ReverseDiffVJP(true)` performs best (this can vary depending on the computer), but often `EnzymeVJP` is the better choice. Generally, `GaussAdjoint` with `EnzymeVJP` is often the best combination, but as seen above, this is not always the case. Therefore, for larger models where runtime can be substantial, we recommend benchmarking different adjoint algorithms and VJP methods to find the best configuration for your specific problem.

Lastly, it should be noted that even if `gradient_method=:Adjoint` is the fastest option for larger models, we still recommend using `:ForwardDiff` if it is not substantially slower. This is because computing the gradient via adjoint methods is much more challenging than with forward methods, as the adjoint approach requires solving a difficult adjoint ODE. In our benchmarks, we have observed that sometimes `:ForwardDiff` successfully computes the gradient, while `:Adjoint` does not. Moreover, forward methods tend to produce more accurate gradients.

## References

```@bibliography
Pages = ["Bachmann.md"]
Canonical = false
```
