# Bayesian Inference

When performing parameter estimation for a model with PEtab.jl, the unknown model parameters are estimated within a frequentist framework, where the goal is to find the maximum likelihood estimate. When prior knowledge about the parameters is available, Bayesian inference offers an alternative approach to fitting a model to data. The aim of Bayesian inference is to infer the posterior distribution of unknown parameters given the data, $\pi(\mathbf{x} \mid \mathbf{y})$, by running a Markov chain Monte Carlo (MCMC) algorithm to sample from the posterior. A major challenge, aside from creating a good model, is to effectively sample the posterior. PEtab.jl supports Bayesian inference via two packages that implement different sampling algorithms:

- **Adaptive Metropolis Hastings Samplers** available in [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl) [vihola2014ergonomic](@cite).
- **Hamiltonian Monte Carlo (HMC) Samplers** available in [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl). The default HMC sampler is the NUTS sampler, which is the default in Stan [hoffman2014no, carpenter2017stan](@cite). HMC samplers are often efficient for continuous targets (models with non-discrete parameters).

This tutorial covers how to create a `PEtabODEProblem` with priors and how to use [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) for Bayesian inference. It should be noted that this part of PEtab.jl is planned to be moved to a separate package, so the syntax will change and be made more user-friendly in the future.

!!! note
    To use the Bayesian inference functionality in PEtab.jl, the Bijectors.jl, LogDensityProblems.jl, and LogDensityProblemsAD.jl packages must be loaded.

## Creating a Bayesian Inference Problem

If a PEtab problem is in the PEtab standard format, priors are defined in the [parameter table](https://petab.readthedocs.io/en/latest/documentation_data_format.html#parameter-table). Here, we focus on the case when the model is defined directly in Julia, using a simple saturated growth model. First, we create the model and simulate some data:

```@example 1
using Distributions, ModelingToolkit, OrdinaryDiffEq, Plots
using ModelingToolkit: D_nounits as D
@mtkmodel SYS begin
    @parameters begin
        b1
        b2
    end
    @variables begin
        x(t) = 0.0
    end
    @equations begin
        D(x) ~ b2 * (b1 - x)
    end
end
@mtkbuild sys = SYS()

# Simulate data with normal measurement noise and σ = 0.03
import Random # hide
Random.seed!(1234) # hide
oprob = ODEProblem(sys, [0.0], (0.0, 2.5), [1.0, 0.2])
tsave = range(0.0, 2.5, 101)
dist = Normal(0.0, 0.03)
sol = solve(oprob, Rodas4(), abstol=1e-12, reltol=1e-12, saveat=tsave)
obs = sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=2.0) # hide
plot(sol.t, obs, seriestype=:scatter, title = "Observed data")
```

Given this, we can now create a `PEtabODEProblem` (for an introduction, see the starting [tutorial](@ref tutorial)):

```@example 1
using DataFrames, PEtab
measurements = DataFrame(obs_id="obs_X", time=sol.t, measurement=obs)
@parameters sigma
obs_X = PEtabObservable(:x, sigma)
observables = Dict("obs_X" => obs_X)
nothing # hide
```

When defining parameters to estimate via `PEtabParameter`, a prior can be assigned using any continuous distribution available in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For instance, we can set the following priors:

- `b_1`: Uniform distribution between 0.0 and 5.0; `Uniform(0.0, 5.0)`.
- `log10_b2`: Uniform distribution between -6.0 and log10(5.0); `Uniform(-6.0, log10(5.0))`.
- `sigma`: Gamma distribution with shape and rate parameters both set to 1.0, `Gamma(1.0, 1.0)`.

Using the following code:

```@example 1
p_b1 = PEtabParameter(:b1, value=1.0, lb=0.0, ub=5.0, scale=:log10, prior_on_linear_scale=true, prior=Uniform(0.0, 5.0))
p_b2 = PEtabParameter(:b2, value=0.2, scale=:log10, prior_on_linear_scale=false, prior=Uniform(-6, log10(5.0)))
p_sigma = PEtabParameter(:sigma, value=0.03, lb=1e-3, ub=1e2, scale=:lin, prior_on_linear_scale=true, prior=Gamma(1.0, 1.0))
pest = [p_b1, p_b2, p_sigma]
```

When specifying priors, it is important to keep in mind the parameter scale (where `log10` is the default). In particular, when `prior_on_linear_scale=false`, the prior applies to the parameter scale, so for `b2` above, the prior is on the `log10` scale. If `prior_on_linear_scale=true` (the default), the prior is on the linear scale, which applies to `b1` and `sigma` above. If a prior is not specified, the default prior is a Uniform distribution on the parameter scale, with bounds corresponding to the upper and lower bounds specified for the `PEtabParameter`. With these priors, we can now create the `PEtabODEProblem`.

```@example 1
osolver = ODESolver(Rodas5P(), abstol=1e-6, reltol=1e-6)
model = PEtabModel(sys, observables, measurements, pest)
petab_prob = PEtabODEProblem(model; odesolver=osolver)
```

## Bayesian Inference (General Setup)

The first step in in order to run Bayesian inference is to construct a `PEtabLogDensity`. This structure supports the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface, meaning it contains all the necessary methods for running Bayesian inference:

```@example 1
using Bijectors, LogDensityProblems, LogDensityProblemsAD
target = PEtabLogDensity(petab_prob)
```

When performing Bayesian inference, the settings for the ODE solver and gradient computations are those specified in `petab_prob`. In this case, we use the default gradient method (`ForwardDiff`) and simulate the ODE model using the `Rodas5P` ODE solver.

One important consideration before running Bayesian inference is the starting point. For simplicity, we here use the parameter vector that was used for simulating the data, but note that typically inference should be performed using at least four chains from different starting points [gelman2020bayesian](@cite):

```@example 1
x = get_x(petab_prob)
nothing # hide
```

Lastly, when performing Bayesian inference with PEtab.jl, it is **important** to note that inference is performed on the prior scale. For instance, if a parameter has `scale=:log10`, but the prior is defined on the linear scale (`prior_on_linear_scale=true`), inference is performed on the linear scale. Additionally, Bayesian inference algorithms typically prefer to operate in an unconstrained space, so a bounded prior like `Uniform(0.0, 5.0)` is not ideal. To address this, bounded parameters are [transformed](https://mc-stan.org/docs/reference-manual/change-of-variables.html) to be unconstrained.

In summary, for a parameter vector on the PEtab parameter scale (`x`), for inference we must transform to the prior scale (`xprior`), and then to the inference scale (`xinference`). This can be done via:

```@example 1
xprior = to_prior_scale(petab_prob.xnominal_transformed, target)
xinference = target.inference_info.bijectors(xprior)
```

!!! warn
    To get correct inference results, it is important that the starting value is on the transformed parameter scale (as `xinference` above).

## Bayesian inference with AdvancedHMC.jl (NUTS)

Given a starting point we can run the NUTS sampler with 2000 samples, and 1000 adaptation steps:

```@example 1
using AdvancedHMC
# δ=0.8 - acceptance rate (default in Stan)
sampler = NUTS(0.8)
Random.seed!(1234) # hide
res = sample(target, sampler, 2000; n_adapts = 1000, initial_params = xinference, 
             drop_warmup=true, progress=false)
nothing #hide
```

Any other algorithm found in AdvancedHMC.jl [documentation](https://github.com/TuringLang/AdvancedHMC.jl) can also be used. To get the output in an easy to interact with format, we can convert it to a [MCMCChains](https://github.com/TuringLang/MCMCChains.jl)

```@example 1
using MCMCChains
chain_hmc = PEtab.to_chains(res, target)
```

which we can also plot:

```@example 1
using Plots, StatsPlots
plot(chain_hmc)
```

!!! note
    When converting the output to a `MCMCChains` the parameters are transformed to the prior-scale (inference scale).

## Bayesian inference with AdaptiveMCMC.jl

Given a starting point we can run the robust adaptive MCMC sampler for $100 \, 000$ iterations with:

```@example 1
using AdaptiveMCMC
Random.seed!(123) # hide
# target.logtarget = posterior logdensity
res = adaptive_rwm(xinference, target.logtarget, 100000; progress=false)
nothing #hide
```

and we can convert the output to a `MCMCChains`

```@example 1
chain_adapt = to_chains(res, target)
plot(chain_adapt)
```

Any other algorithm found in AdaptiveMCMC.jl [documentation](https://github.com/mvihola/AdaptiveMCMC.jl) can also be used.

## References

```@bibliography
Pages = ["inference.md"]
Canonical = false
```
