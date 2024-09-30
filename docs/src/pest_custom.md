# [Wrapping Optimization Packages](@id wrap_est)

A `PEtabODEProblem` contains all the necessary information for wrapping a suitable optimizer to estimate model parameters. Since wrapping a package can be cumbersome, PEtab.jl provides wrappers for performing single-start parameter estimation (with [`calibrate`](@ref)) and multi-start parameter estimation (with [`calibrate_multistart`](@ref)). More details can be found in [this](@ref pest_methods) tutorial. Still, in some cases, it may be necessary to manually wrap one of the optimization packages not supported by PEtab.jl.

This tutorial show how to wrap an existing optimization package, using the `IPNewton` method from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) as an example. As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial](@ref tutorial). Even though the code below provides the model as a `ReactionSystem`, everything works exactly the same if the model is provided as an `ODESystem`.

```@example 1
using Catalyst, PEtab

# Create the dynamic model
t = default_t()
rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]

# Observables
@unpack E, S = rn
obs_sum = PEtabObservable(S + E, 3.0)
@unpack P = rn
@parameters sigma
obs_p = PEtabObservable(P, sigma)
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)

# Parameters to estimate
p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

# Simulate measurement data with 'true' parameters
using OrdinaryDiffEq, DataFrames
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs_sum = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs_p = sol[:P] + .+ randn(length(sol[:P]))
df_sum = DataFrame(obs_id = "obs_sum", time = sol.t, measurement = obs_sum)
df_p = DataFrame(obs_id = "obs_p", time = sol.t, measurement = obs_p)
measurements = vcat(df_sum, df_p)

model = PEtabModel(rn, observables, measurements, pest; speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

## Extracting Relevant Input from a PEtabODEProblem

A numerical optimizer requires an objective function, and derivative-based methods also need a gradient function and, in some cases, a Hessian function. Following the [PEtab standard](https://petab.readthedocs.io/en/latest/), PEtab.jl works with likelihoods, so the objective function corresponds to the negative log-likelihood (`nllh`), which can be accessed with:

```@example 1
x = get_x(petab_prob)
nllh = petab_prob.nllh(x; prior = true)
```

Here, the keyword argument `prior = true` (default) ensures that potential parameter priors are considered when computing the likelihood. Furthermore, the `PEtabODEProblem` provides both in-place and out-of-place gradient functions:

```@example 1
g_inplace = similar(x)
petab_prob.grad!(g_inplace, x; prior = true)
g_outplace = petab_prob.grad(x)
```

as well as in-place and out-of-place Hessian functions:

```@example 1
h_inplace = zeros(length(x), length(x))
petab_prob.hess!(h_inplace, x; prior = true)
h_outplace = petab_prob.hess(x)
```

In the above cases, the input parameter vector is a `ComponentArray`, but a `Vector` input is also accepted, and in this case, the gradient functions will also output a `Vector`. Additionally, the gradients and Hessians are computed using the default methods in the `PEtabODEProblem` (for more details, see [this](@ref default_options) page).

Lastly, for parameter estimation with ODE models, it is often useful to set parameter bounds. Because, without bounds, the optimization algorithm can explore regions where the ODE solver fails to solve the model which prolongs runtime [frohlich2022fides](@cite). The bounds can be accessed via:

```@example 1
lb, ub = petab_prob.lower_bounds, petab_prob.upper_bounds
nothing # hide
```

Both `lb` and `ub` are `ComponentArray`s. If an optimization package does not support `ComponentArray` (as in the example below), they can be converted to a `Vector` by calling `collect`.

## Wrapping Optim.jl IPNewton

From the Optim.jl [documentation](https://julianlsolvers.github.io/Optim.jl/stable/), we can see that in order to use the `IPNewton` method, we need to provide the objective, gradient, Hessian, and parameter bounds, where the latter are provided as vectors. Using the information outlined above, we can do:

```@example 1
using Optim
x0 = collect(get_x(petab_prob))
df = TwiceDifferentiable(petab_prob.nllh, petab_prob.grad!, petab_prob.hess!, x0)
dfc = TwiceDifferentiableConstraints(collect(lb), collect(ub))
nothing # hide
```

Note that we convert any `ComponentArray` to a `Vector` with `collect`. Given this, we can perform parameter estimation with `x0` as the starting point:

```@example 1
res = Optim.optimize(df, dfc, x0, IPNewton())
```

## References

```@bibliography
Pages = ["pest_custom.md"]
Canonical = false
```
