# Bayesian Inference

When fitting a model with PEtab.jl the unknown model parameters are estimated within a frequentist framework, and the goal is to find the maximum likelihood estimate. When prior knowledge about the parameters is available, Bayesian inference is an alternative approach to fitting the model to data. The aim with Bayesian inference is to infer the posterior distribution of unknown parameters given the data, ``p(\theta | y)`` by running Markov chain Monte Carlo (MCMC) algorithm that samples from the Posterior.

PEtab.jl supports Bayesian inference via two packages:

- **Adaptive Metropolis Hastings Samplers** available in [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl)
- **Hamiltonian Monte Carlo (HMC) Samplers**: available in [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl). Here the default choice is the NUTS sampler, which is used by [Turing.jl](https://github.com/TuringLang/Turing.jl), and is also the default in Stan. HMC samplers are often more efficient than other methods.

This document covers how to create a `PEtabODEProblem` with priors, and how to use both [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) for Bayesian inference.

!!! note
    To use the Bayesian inference functionality in PEtab.jl, the Bijectors, LogDensityProblems, and LogDensityProblemsAD packages must be loaded.

## Setting up a Bayesian inference problems

If the PEtab problem is in the PEtab standard format, priors for model parameters can be defined in the [Parameter table](https://petab.readthedocs.io/en/latest/documentation_data_format.html#parameter-table). Here, we show how to set up priors for a `PEtabODEProblem` defined directly in Julia, using a simple saturated growth model as an example. First, we create the model and simulate some data:

```@example 1; ansicolor=false
using ModelingToolkit, OrdinaryDiffEq, PEtab, Plots

# Dynamic model
@parameters b1, b2
@variables t x(t)
D = Differential(t)
eqs = [
    D(x) ~ b2*(b1 - x)
]
specie_map = [x => 0]
parameter_map = [b1 => 1.0, b2 => 0.2]
@named sys = ODESystem(eqs)

# Simulate data
using Random, Distributions
Random.seed!(1234)
oprob = ODEProblem(sys, specie_map, (0.0, 2.5), parameter_map)
tsave = collect(range(0.0, 2.5, 101))
dist = Normal(0.0, 0.03)
_sol = solve(oprob, Rodas4(), abstol=1e-12, reltol=1e-12, saveat=tsave, tstops=tsave)
# Normal measurement noise with σ = 0.03
obs = _sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm) # hide
plot(_sol.t, obs, seriestype=:scatter, title = "Observed data")
```

Now we can provide the rest of information needed for setting up a `PEtabODEProblem` (for a starter on setting up a `PEtabODEProblem` see [here](@ref define_in_julia)):

```@example 1; ansicolor=false
using DataFrames
# Measurement data
measurements = DataFrame(
    obs_id="obs_X",
    time=_sol.t,
    measurement=obs)
# Observable
@parameters sigma
obs_X = PEtabObservable(x, sigma)
observables = Dict("obs_X" => obs_X)
```

When defining the parameters to infer, we can assign a prior using any continuous distribution available in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For instance, we can set the following priors:

- ``b_1``: Uniform distribution between 0.0 and 5.0; ``b_1 \sim \mathcal{U}(0.0, 5.0)``.
- ``\mathrm{log}_{10}(b_2)`` : Uniform distribution between -6.0 and ``\mathrm{log}_{10}(5.0)``, ``\mathrm{log}_{10}(b_2) \sim \mathcal{U}\big(-6.0, \mathrm{log}_{10}(5.0) \big)``.
- ``\sigma`` : Gamma distribution with shape and rate parameters both set to 1.0, ``\sigma \sim \mathcal{G}(1.0, 1.0)``.

```@example 1; ansicolor=false
_b1 = PEtabParameter(b1, value=1.0, lb=0.0, ub=5.0, scale=:log10, prior_on_linear_scale=true, prior=Uniform(0.0, 5.0))
_b2 = PEtabParameter(b2, value=0.2, scale=:log10, prior_on_linear_scale=false, prior=Uniform(-6, log10(5.0)))
_sigma = PEtabParameter(sigma, value=0.03, lb=1e-3, ub=1e2, scale=:lin, prior_on_linear_scale=true, prior=Gamma(1.0, 1.0))
parameters_ets = [_b1, _b2, _sigma]
```

When specifying priors in PEtab.jl, it is important to note that parameters are by default estimated on the $\mathrm{log}_{10}$ scale (can be changed by `scale` argument). When `prior_on_linear_scale=false` the prior applies to this parameter scale (default $\mathrm{log}_{10}$), therefore the prior for `b2` is on the $\mathrm{log}_{10}$ scale. If `prior_on_linear_scale=true`, the prior is in the linear scale, which in this case holds for `b1` and `sigma`. If a prior is not specified, the default prior is a Uniform distribution on the parameter scale, with bounds that correspond to upper and lower bounds specified for the `PEtabParameter`.

With the priors defined, we can proceed to create a `PEtabODEProblem`.

```@example 1; ansicolor=false
petab_model = PEtabModel(sys, observables, measurements, parameters_ets; state_map=specie_map, verbose=false)
petab_problem = PEtabODEProblem(petab_model; ode_solver=ODESolver(Rodas5(), abstol=1e-6, reltol=1e-6), verbose=false)
nothing #hide
```

## Bayesian inference (general setup)

The first step to performing Bayesian inference is to construct a `PEtabLogDensity`. This structure supports the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface, meaning it supports all the necessary methods for performing Bayesian inference. To create it provide the `PEtabODEProblem` as an argument.

```@example 1; ansicolor=false
using Bijectors, LogDensityProblems, LogDensityProblemsAD
target = PEtabLogDensity(petab_problem)
```

When later performing Bayesian inference, the settings for the ODE solver and gradient computations are those in `petab_problem`. For instance, in this case, we use the default gradient method (`ForwardDiff`) and simulate the ODE model with the `Rodas5` ODE solver.

Inference can now be performed. Although the choice of a starting point for the inference process is crucial, for simplicity we use the parameter vector used for simulating the data (note that the second parameter is on $\mathrm{log}_{10}$ scale).

```@example 1; ansicolor=false
xpetab = petab_problem.θ_nominalT
```

Lastly, when conducting Bayesian inference with PEtab.jl, an **important** note is that inference is performed on the prior scale. For instance, if a parameter is set with `scale=:log10`, but the prior is defined on the linear scale (`prior_on_linear_scale=true`), then inference is performed on the linear scale. Moreover, Bayesian inference algorithms typically prefer to operate in an unconstrained space, that is a prior such as $b_1 \sim \mathcal{U}(0.0, 5.0)$, where the parameter is bounded is not ideal. To address this, bounded parameters are [transformed](https://mc-stan.org/docs/reference-manual/change-of-variables.html) to be unconstrained.

In summary, for a parameter vector on the PEtab parameter scale (`xpetab`), for inference we must transform to the prior scale (`xprior`), and then to the inference scale (`xinference`). This can be done via:

```@example 1; ansicolor=false
xprior = to_prior_scale(petab_problem.θ_nominalT, target)
xinference = target.inference_info.bijectors(xprior)
```

!!! warn
    To get correct inference results, it is important that the starting value is on the transformed parameter scale (as `xinference` above).

## Bayesian inference with AdvancedHMC.jl (NUTS)

Given a starting point we can run the NUTS sampler with 2000 samples, and 1000 adaptation steps:

```@example 1; ansicolor=false
using AdvancedHMC
# δ=0.8 - acceptance rate (default in Stan)
sampler = NUTS(0.8)
Random.seed!(1234)
res = sample(target, sampler,
             2000;
             nadapts = 1000,
             initial_params = xinference,
             drop_warmup=true,
             progress=false)
nothing #hide
```

Any other algorithm found in AdvancedHMC.jl [documentation](https://github.com/TuringLang/AdvancedHMC.jl) can also be used.

To put the output into an easy to interact with format, we can convert it to a [MCMCChains](https://github.com/TuringLang/MCMCChains.jl)

```@example 1; ansicolor=false
using MCMCChains
chain_hmc = PEtab.to_chains(res, target)
```

which we can also plot:

```@example 1; ansicolor=false
using Plots, StatsPlots
plot(chain_hmc)
```

!!! note
    When converting the output to a `MCMCChains` the parameters are transformed to the prior-scale (inference scale).

## Bayesian inference with AdaptiveMCMC.jl (NUTS)

Given a starting point we can run the robust adaptive MCMC sampler with $200 \, 000$ by:

```@example 1; ansicolor=false
using AdaptiveMCMC
Random.seed!(123)
# target.logtarget = posterior logdensity
res = adaptive_rwm(xinference, target.logtarget, 200000; progress=false)
nothing #hide
```

and we can convert the output to a `MCMCChains`

```@example 1; ansicolor=false
chain_adapt = to_chains(res, target)
plot(chain_adapt)
```

Any other algorithm found in AdaptiveMCMC.jl [documentation](https://github.com/mvihola/AdaptiveMCMC.jl) can also be used.
