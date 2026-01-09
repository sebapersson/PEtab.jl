# [Bayesian inference](@id bayesian_inference)

PEtab.jl’s parameter-estimation workflow is frequentist and targets a maximum-likelihood (or
maximum a posteriori, when priors are included) estimate. When prior knowledge about the
parameters is available, Bayesian inference offers an alternative approach to model fitting
by inferring the posterior distribution of the parameters given data,
$\pi(\mathbf{x}\mid \mathbf{y})$, typically via Markov chain Monte Carlo (MCMC) sampling.
PEtab.jl supports Bayesian inference through two sampler families:

- **Adaptive Metropolis–Hastings** samplers from
  [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl)
  [vihola2014ergonomic](@cite).
- **Hamiltonian Monte Carlo (HMC)** samplers from
  [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl), including NUTS as in Stan
  [hoffman2014no, carpenter2017stan](@cite).

This tutorial shows how to (i) define priors in a `PEtabODEProblem`, and (ii) run Bayesian
inference with AdaptiveMCMC.jl and AdvancedHMC.jl. Note that this functionality is planned
to move into a separate package, and the API may change.

!!! note To use Bayesian inference functionality, load Bijectors.jl, LogDensityProblems.jl,
and LogDensityProblemsAD.jl.

## Creating a Bayesian inference problem

For PEtab problems in the PEtab standard format, priors are defined in the
[parameter table](https://petab.readthedocs.io/en/latest/documentation_data_format.html#parameter-table).
Here, we instead consider a model defined directly in Julia: a simple saturated growth
model. First, let’s define the model and simulate data:

```@example 1
using Distributions, ModelingToolkit, Plots
using DataFrames, PEtab
using ModelingToolkit: t_nounits as t, D_nounits as D

@mtkmodel SYS begin
    @parameters begin
        b1
        b2
    end
    @variables begin
        x(t) = 0.0
        # observables
        obs_x(t)
    end
    @equations begin
        D(x) ~ b2 * (b1 - x)
        # observables
        obs_x ~ x
    end
end
@mtkbuild sys = SYS()

# Simulate data with Normal measurement noise (σ = 0.03)
import Random # hide
Random.seed!(1234) # hide
oprob = ODEProblem(sys, [0.0], (0.0, 2.5), [1.0, 0.2])
tsave = range(0.0, 2.5, 101)
sol = solve(oprob, Rodas4(); abstol=1e-12, reltol=1e-12, saveat=tsave)
obs = sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))
measurements = DataFrame(obs_id="obs_x", time=sol.t, measurement=obs)

# Observable
@parameters sigma
observables = PEtabObservable(:obs_x, :obs_x, sigma)

# Plot the data
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=2.0) # hide
plot(measurements.time, measurements.measurement; seriestype=:scatter,
    title="Observed data")
```

Priors are assigned via `PEtabParameter` using any continuous distribution from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For example:

```@example 1
p_b1 = PEtabParameter(:b1, value=1.0, scale=:lin, prior=Uniform(0.0, 5.0))
p_b2 = PEtabParameter(:b2, value=0.2, prior=LogNormal(1.0, 1.0))
p_sigma = PEtabParameter(:sigma, value=0.03, scale=:lin, prior=Gamma(1.0, 1.0))
pest = [p_b1, p_b2, p_sigma]
```

Priors are evaluated on the parameter’s **linear** (non-transformed) scale. For example,
while `b2` above is estimated on the `log10` scale, the prior applies to `b1` (not
`log10(b1)`). If no prior is provided, the default is a `Uniform` over the parameter bounds.
Finally, build the problem as usual:

```@example 1
model = PEtabModel(sys, observables, measurements, pest)
petab_prob = PEtabODEProblem(model)
```

## Bayesian inference (general setup)

To run Bayesian inference, first construct a `PEtabLogDensity`. This object implements the
[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface and can be
used as the target density for MCMC:

```@example 1
using Bijectors, LogDensityProblems, LogDensityProblemsAD
target = PEtabLogDensity(petab_prob)
```

ODE-solver and derivative settings are taken from `petab_prob` (here, the default gradient
method `:ForwardDiff` and the `Rodas5P` solver).

A key choice is the starting point. For simplicity, we start from the nominal parameter
vector, but in practice inference should be run with multiple chains and dispersed starting
points [gelman2020bayesian](@cite):

```@example 1
x = get_x(petab_prob)
nothing # hide
```

It is important to note inference in PEtab.jl is performed on an **linear** parameter scale.
Therefore, to run inference parameter on transformed scale (e.g. `scale = :log10`) must
first mapped back to the linear (prior) scale. Moreover, since many samplers operate in an
unconstrained space, bounded priors (e.g. `Uniform(0.0, 5.0)`) parameters need to be are
transformed to $\mathbb{R}$ via bijectors. In short, for a PEtab parameter vector `x`,
inference uses the composition `x -> xprior -> xinference`:

```@example 1
xprior = to_prior_scale(petab_prob.xnominal_transformed, target)
xinference = target.inference_info.bijectors(xprior)
```

!!! warning The initial value passed to the sampler must be on the inference scale
(`xinference`).

## Bayesian inference with AdvancedHMC.jl (NUTS)

Given a starting point, NUTS can be run with 2000 samples and 1000 adaptation steps via:

```@example 1
using AdvancedHMC
# δ = 0.8 is the target acceptance rate (default in Stan)
sampler = NUTS(0.8)
Random.seed!(1234) # hide
res = sample(target, sampler, 2000;
    n_adapts = 1000,
    initial_params = xinference,
    drop_warmup = true,
    progress = false,
)
nothing # hide
```

Any sampler supported by AdvancedHMC.jl can be used (see the
[documentation](https://github.com/TuringLang/AdvancedHMC.jl)). To work with results using a
standard interface, convert to an `MCMCChains.Chains`:

```@example 1
using MCMCChains
chain_hmc = PEtab.to_chains(res, target)
```

This can be plotted with:

```@example 1
using Plots, StatsPlots
plot(chain_hmc)
```

!!! note `PEtab.to_chains` converts samples back to the **prior (linear) scale** (not the
unconstrained inference scale).

## Bayesian inference with AdaptiveMCMC.jl

Given a starting point, the robst adaptive random-walk Metropolis sampler can be run for
$100\,000$ samples with:

```@example 1
using AdaptiveMCMC
Random.seed!(123) # hide
# target.logtarget is the posterior log density on the inference scale
res = adaptive_rwm(xinference, target.logtarget, 100000; progress=false)
nothing # hide
```

Convert the result to an `MCMCChains.Chains`:

```@example 1
using MCMCChains, Plots, StatsPlots
chain_adapt = PEtab.to_chains(res, target)
plot(chain_adapt)
```

Other samplers from AdaptiveMCMC.jl can be used; see the
[documentation](https://github.com/mvihola/AdaptiveMCMC.jl).

## References

```@bibliography
Pages = ["inference.md"]
Canonical = false
```
