# [Speeding up condition-specific parameters](@id Beer_tut)

Some models include parameters whose values differ across simulation conditions. For such
problems, when gradients are computed with automatic differentiation
(`gradient_method = :ForwardDiff` or `gradient_method = :ForwardEquations`), runtime can
often be reduced substantially by setting `split_over_conditions = true` when constructing a
`PEtabODEProblem`.

This page explains when and why this helps, and assumes familiarity with condition-specific
parameters (see [Simulation condition-specific parameters](@ref condition_parameters)). As a
working example, a published model is used (the Beer model [beer2014creating](@cite)),
available in the PEtab standard format (see [Importing PEtab problems](@ref
import_petab_problem)). Given the PEtab files (downloadable from
[here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/beer)), the problem
can be imported with:

```@example 1
using PEtab
# path_yaml depends on where the problem is saved
path_yaml = joinpath(@__DIR__, "assets", "beer", "Beer_MolBioSystems2014.yaml")
model = PEtabModel(path_yaml)
nothing # hide
```

## Efficient handling of condition-specific parameters

The Beer problem has 4 species and 9 ODE parameters, but 72 parameters are estimated because
many parameters are condition-specific. For example, `cond1` has `τ_cond1` and `cond2` has
`τ_cond2`, which both map to the ODE parameter `τ`. This is reflected by the model
statistics are:

```@example 1
using Catalyst
petab_prob = PEtabODEProblem(model)
println("Number of ODE model species      = ", length(unknowns(model.sys)))
println("Number of ODE model parameters   = ", length(parameters(model.sys)))
println("Number of parameters to estimate = ", length(petab_prob.xnames))
```

For small ODE systems, `gradient_method = :ForwardDiff` is typically fastest, and
`hessian_method = :ForwardDiff` is often feasible (see [Derivative methods (gradients and
Hessians)](@ref gradient_support)). By default, PEtab.jl computes derivatives with a single
`ForwardDiff.gradient` call over all simulation conditions. For condition-specific
parameters this can be wasteful: if `n` directional passes are needed for the full parameter
vector, condition `i` may only require `n_i < n` passes because many parameters are inactive
in that condition.

To reduce this overhead, `split_over_conditions = true` computes derivatives per condition
(one ForwardDiff call per simulation condition). Here, the effect on gradient runtime is
noticeable:

```@example 1
using Printf
petab_prob1 = PEtabODEProblem(model; split_over_conditions = true)
petab_prob2 = PEtabODEProblem(model; split_over_conditions = false)
x = get_x(petab_prob1)
g1, g2 = similar(x), similar(x)
petab_prob1.grad!(g1, x) # hide
petab_prob2.grad!(g2, x) # hide
b1 = @elapsed petab_prob1.grad!(g1, x)
b2 = @elapsed petab_prob2.grad!(g2, x)
@printf("Runtime split_over_conditions = true: %.2fs\n", b1)
@printf("Runtime split_over_conditions = false: %.2fs\n", b2)
```

For the Hessian, the difference is typically even larger:

```@example 1
h1, h2 = zeros(length(x), length(x)), zeros(length(x), length(x))
_ = petab_prob1.nllh(x) # hide
_ = petab_prob2.nllh(x) # hide
petab_prob1.hess!(h1, x) # hide
petab_prob2.hess!(h2, x) # hide
b1 = @elapsed petab_prob1.hess!(h1, x)
b2 = @elapsed petab_prob2.hess!(h2, x)
@printf("Runtime split_over_conditions = true: %.1fs\n", b1)
@printf("Runtime split_over_conditions = false: %.1fs\n", b2)
```

A natural question is why `split_over_conditions = true` is not always the default. The
reason is overhead: evaluating ForwardDiff separately per condition can be slower when there
are few or no condition-specific parameters due to overhead. PEtab.jl therefore uses a
heuristic and enables `split_over_conditions` by default when the number of
condition-specific parameters is at least twice the number of ODE parameters. This heuristic
is conservative, so for models with many condition-specific parameters it is recommended to
benchmark `split_over_conditions = true` vs `false`.

## References

```@bibliography
Pages = ["Beer.md"]
Canonical = false
```
