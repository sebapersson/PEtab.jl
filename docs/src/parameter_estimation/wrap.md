# [Using optimizers directly](@id wrap_est)

A `PEtabODEProblem` contains all information needed to use an optimizer directly for
parameter estimation. While PEtab.jl provides high-level functions for single-start
estimation ([`calibrate`](@ref)) and multi-start estimation
([`calibrate_multistart`](@ref)), direct use can be useful for unsupported optimization
packages or custom workflows.

This tutorial demonstrates this approach using the `IPNewton` method from
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). As a running example, the
Michaelis–Menten model from the [starting tutorial](@ref tutorial) is used.

```@example 1
using Catalyst, PEtab
rn = @reaction_network begin
    @parameters begin
      S0
      c3 = 1.0
    end
    @species begin
      S(t) = S0
      E(t) = 50.0
      P(t) = 0.0
    end
    @observables begin
      obs1 ~ S + E
      obs2 ~ P
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Observables
@parameters sigma
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
observables = [petab_obs1, petab_obs2]

# Parameters to estimate
p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

# Measurements; simulate with 'true' parameters
using DataFrames, OrdinaryDiffEqRosenbrock
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs1 = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs2   = sol[:P] .+ randn(length(sol[:P]))
df1 = DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1)
df2 = DataFrame(obs_id = "petab_obs2", time = sol.t, measurement = obs2)
measurements = vcat(df1, df2)

# Create the PEtabODEProblem
petab_model = PEtabModel(rn, observables, measurements, pest)
petab_prob = PEtabODEProblem(petab_model)
```

## Extracting relevant functions from a PEtabODEProblem

A numerical optimizer requires an objective function, often a gradient function, and
sometimes a Hessian function. PEtab.jl works with likelihoods, so the objective corresponds
to the negative log-likelihood (`nllh`):

```@example 1
x = get_x(petab_prob)
nllh = petab_prob.nllh(x; prior = true)
```

The keyword argument `prior = true` (default) includes parameter priors (if defined). In
addition, the `PEtabODEProblem` provides both in-place and out-of-place gradients:

```@example 1
g_inplace = similar(x)
petab_prob.grad!(g_inplace, x; prior = true)
g_outplace = petab_prob.grad(x; prior = true)
```

and Hessians:

```@example 1
h_inplace = zeros(length(x), length(x))
petab_prob.hess!(h_inplace, x; prior = true)
h_outplace = petab_prob.hess(x; prior = true)
```

The input `x` is typically a `ComponentArray`, but `Vector` inputs are also supported (the
type of output matches the input). More details on what is available in a `PEtabODEProblem`
can be found in the [API documentation](@ref API)

For ODE models, parameter bounds are often important: without bounds, the optimizer may
explore regions where the ODE solver fails, increasing runtime [frohlich2022fides](@cite).
Bounds are available as:

```@example 1
lb, ub = petab_prob.lower_bounds, petab_prob.upper_bounds
nothing # hide
```

`lb` and `ub` are `ComponentArray`s. If an optimizer does not support `ComponentArray`,
convert them to `Vector`s with `collect`; e.g., `collect(lb)`.

## Wrapping Optim.jl `IPNewton`

Optim.jl’s `IPNewton` expects an objective, gradient, Hessian, and box constraints (bounds
provided as plain vectors). Using the functions in a `PEtabODEProblem` (see above), this can
be set up as:

```@example 1
using Optim
x0 = collect(get_x(petab_prob))
df  = TwiceDifferentiable(petab_prob.nllh, petab_prob.grad!, petab_prob.hess!, x0)
dfc = TwiceDifferentiableConstraints(collect(lb), collect(ub))
nothing # hide
```

Here, `collect` converts `ComponentArray`s to `Vector`s for Optim.jl. The optimization can
then be run from `x0`:

```@example 1
opt_res = Optim.optimize(df, dfc, x0, IPNewton())
```

## References

```@bibliography
Pages = ["pest_custom.md"]
Canonical = false
```
