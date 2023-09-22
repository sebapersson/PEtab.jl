# [Models with many conditions specific parameters](@id Beer_tut)

Here we will create a `PEtabODEproblem` for a small ODE-model with many parameters to estimate. Specifically, the ODE-system has $\leq 20$ states and $\leq 20$ parameters, but there are approximately 70 parameters to estimate, since most parameters are specific to a subset of simulation conditions. For example, *cond1* has a parameter τ_cond1, and *cond2* has τ_cond2, which maps to the ODE-system parameter τ, respectively.

To run the code, you need the Beer PEtab files, which you can find [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Beer.jl). You can also find a fully runnable example of this tutorial [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Beer.jl).

First, we load the necessary libraries and read the model.

```julia
using PEtab
using OrdinaryDiffEq
using Printf

path_yaml = joinpath(@__DIR__, "Beer", "Beer_MolBioSystems2014.yaml") 
petab_model = PEtabModel(path_yaml, verbose=true)
```
```
PEtabModel for model Beer. ODE-system has 4 states and 9 parameters.
Generated Julia files are at ...
```

## Handling condition-specific parameters

When dealing with small ODE-systems like Beer, the most efficient gradient method is `gradient_method=:ForwardDiff`. Additionally, we can compute the hessian via `hessian_method=:ForwardDiff`. However, by default, we have to perform as many forward-passes (solve the ODE model) as there are model-parameters when we compute the gradient and hessian as we compute derivatives by a single call to ForwardDiff.jl. This is problematic since the Beer model has both many simulation conditions and model parameters to estimate. To address this issue, we can use the option `split_over_conditions=true` to force one ForwardDiff.jl call per simulation condition. This is most efficient for models where a majority of parameters are specific to a subset of simulation conditions.

For a model like Beer, the following options are thus recommended:

1. `ode_solver` - Rodas5P() (works well for smaller models with up to 15 states) and we use the default `abstol, reltol .= 1e-8`.
2. `gradient_method` - For small models like Beer, forward mode automatic differentiation (AD) is the fastest, so we choose `:ForwardDiff`.
3. `hessian_method` - For small models like Boehm with up to 20 parameters, it is computationally feasible to compute the full Hessian via forward-mode AD. Thus, we choose `:ForwardDiff`.
4. `split_over_conditions=true` - This forces a call to ForwardDiff.jl per simulation condition.

```julia
ode_solver = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8)
petab_problem = PEtabODEProblem(petab_model, ode_solver, 
                                gradient_method=:ForwardDiff, 
                                hessian_method=:ForwardDiff, 
                                split_over_conditions=true, 
                                sparse_jacobian=false)

p = petab_problem.θ_nominalT
gradient = zeros(length(p))
hessian = zeros(length(p), length(p))
cost = petab_problem.compute_cost(p)
petab_problem.compute_gradient!(gradient, p)
petab_problem.compute_hessian!(hessian, p)
@printf("Cost = %.2f\n", cost)
@printf("First element in the gradient = %.2e\n", gradient[1])
@printf("First element in the hessian = %.2f\n", hessian[1, 1])
```
```
Cost = -58622.91
First element in the gradient = 7.17e-02
First element in the hessian = 755266.33
```
