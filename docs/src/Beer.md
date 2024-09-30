# [Condition-Specific Parameters](@id Beer_tut)

As discussed in [this](@ref define_conditions) extended tutorial, sometimes a subset of the model parameters to estimate have different values across experimental/simulation conditions. For such models, runtime can drastically improve by setting `split_over_conditions=true` when creating a `PEtabODEProblem`. This example explores this option in more detail, and it assumes that that you are familiar with condition-specific parameters in PEtab (see [this](@ref define_conditions) tutorial) and with the gradient methods in PEtab.jl (see [this](@ref gradient_support) page).

As a working example, we use a published signaling model referred to as the Beer model after the first author [beer2014creating](@cite). The Beer model is available in the PEtab standard format (a tutorial on importing problems in the standard format can be found [here](@ref import_petab_problem)), and the PEtab files for this model can be downloaded from [here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/beer). Given the problem YAML file, we can import the problem as:

```@example 1
using PEtab
path_yaml = joinpath(@__DIR__, "assets", "beer", "Beer_MolBioSystems2014.yaml")
model = PEtabModel(path_yaml)
nothing # hide
```

## Efficient Handling of Condition-Specific Parameters

The Beer problem is a small model with 4 species and 9 parameters in the ODE system, but there are 72 parameters to estimate. This is because most parameters are specific to a subset of simulation conditions. For example, `cond1` has a parameter `τ_cond1`, and `cond2` has `τ_cond2`, which map to the ODE model parameter `τ`, respectively. This can be seen by printing some model statistics:

```@example 1
using Catalyst
petab_prob = PEtabODEProblem(model)
println("Number of ODE model species = ", length(unknowns(model.sys)))
println("Number of ODE model parameters = ", length(parameters(model.sys)))
println("Number of parameters to estimate = ", length(petab_prob.xnames))
```

For small ODE models like the Beer model, the most efficient gradient method is `gradient_method=:ForwardDiff`, and it is often feasible to compute the Hessian using `hessian_method=:ForwardDiff` as well (see [this](@ref gradient_support) page for details). Typically, with `:ForwardDiff`, PEtab.jl computes the gradient with a single call to `ForwardDiff.gradient`. However, for the Beer model, this approach is problematic because for each simulation condition, `n` forward passes are required to compute all derivatives, where `n` depends on the number of gradient parameters. Since many parameters only belong to a subset of conditions, actually only `ni < n` forward passes are needed for each condition. To this end, PEtab.jl provides the `split_over_conditions=true` keyword when building the `PEtabODEProblem`, which ensures that one `ForwardDiff.gradient` call is performed per simulation condition. Let us examine how this affects gradient runtime for the Beer model:

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

For the Hessian, the difference in runtime is even larger:

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

Given that `split_over_conditions=true` reduces runtime in the example above, a natural question is: why is it not the default option in PEtab.jl? This is because calling `ForwardDiff.gradient` for each simulation condition, instead of once for all conditions, introduces an overhead. Therefore, for models with none or very few condition-specific parameters, `split_over_conditions=false` is faster. Determining exactly how many condition-specific parameters are needed to make `true` the faster option is difficult. Currently, the default is to enable this option when the number of condition-specific parameters is at least twice the number of parameters to estimate in the ODE model. For the Beer model, this means `split_over_conditions=true` is set by default, but this is a rough heuristic. Therefore, for models like these, we recommend benchmarking the two configurations to determine which is fastest.

## References

```@bibliography
Pages = ["Beer.md"]
Canonical = false
```
