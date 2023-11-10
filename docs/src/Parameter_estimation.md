# [Parameter Estimation (Model Calibration)](@id parameter_estimation)

PEtab.jl provides interfaces to three optimization packages:

- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/): Supports LBFGS, BFGS, or IPNewton methods.
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/): An interior-point optimizer.
- [Fides](https://github.com/fides-dev/fides): A Newton trust region method.

You can find available options for each optimizer in the [Available Optimizers](@ref options_optimizers) section. To help you choose the right optimizer, based on extensive benchmarks we recomend:

- If you have access to a full Hessian matrix, the Interior-point Newton method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) typically outperforms the trust-region method in Fides.py.
- If you can only provide a Gauss-Newton Hessian approximation (not the full Hessian), the Newton trust-region method in Fides.py is usually more effective than the interior-point method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

!!! note
    Keep in mind that each problem is unique, and although the suggested options are generally effective, they may not always be the ideal choice for a particular model.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`. To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).

Additionally, the `PEtabODEProblem` contain all the necessary information to use other optimization libraries like [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl).

## Multi-Start Local Optimization

A widely adopted and effective approach for model calibration is multi-start local optimization. In this method, a local optimizer is run from a large number (typically 100-1000) of randomly generated initial parameter guesses. These guesses are efficiently generated using techniques like Latin-hypercube sampling to effectively explore the parameter space.

To perform multi-start parameter estimation, you can employ the `calibrate_model_multistart` function. This function requires a `PEtabODEProblem`, the number of multi-starts, one of the available optimizer algorithms, and a directory to save the results. If you provide `dir_save=nothing` as the directory path, the results will not be written to disk. However, as a precaution against premature termination, we strongly recommended to specify a directory.

For example, to use the Interior-point Newton method from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to perform parameter estimation on the Boehm model with 10 multi-starts you can write:

```julia
using PEtab
using Optim

dir_save = joinpath(@__DIR__, "Boehm_opt")
petab_model = PEtabModel(path_to_Boehm_model)
petab_problem = PEtabODEProblem(petab_model)
res = calibrate_model_multistart(petab_problem, IPNewton(), 10, dir_save,
                               options=Optim.Options(iterations = 200))
print(res)
```
```
PEtabMultistartOptimisationResult
--------- Summary ---------
min(f)                = 1.48e+02
Parameters esimtated  = 9
Number of multistarts = 10
Optimiser algorithm   = Optim_IPNewton
```

In this example, we use `Optim.Options` to set the maximum number of iterations to 200. You can find a full list of options [here](https://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/). The results are returned as a `PEtabMultistartOptimisationResult`, which contains the best-found minima (`xmin`), the smallest objective value (`fmin`), and optimization results for each run. In case a `dir_save` is provided results can also easily be read from disk into a `PEtabMultistartOptimisationResult` struct:

```julia
res_read = PEtabMultistartOptimisationResult(dir_save)
print(res_read)
```
```
PEtabMultistartOptimisationResult
--------- Summary ---------
min(f)                = 1.48e+02
Parameters esimtated  = 9
Number of multistarts = 10
Optimiser algorithm   = Optim_IPNewton
```

The method for generating initial parameter guesses can also be chosen, as we support any method available in [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl). For instance, to use Latin-Hypercube sampling (which is the default), and to perform multi-start calibration with Fides, write:

```julia
using PEtab
using PyCall
using QuasiMonteCarlo

petab_model = PEtabModel(path_yaml)
petab_problem = PEtabODEProblem(petab_model)
res = calibrate_model_multistart(petab_problem, Fides(nothing), 10, dir_save,
                               sampling_method=QuasiMonteCarlo.LatinHypercubeSample())
print(res)
```
```
PEtabMultistartOptimisationResult
--------- Summary ---------
min(f)                = 1.38e+02
Parameters esimtated  = 9
Number of multistarts = 10
Optimiser algorithm   = Fides
```

In this example, we utilize Latin-Hypercube sampling by specifying `sampling_method=QuasiMonteCarlo.LatinHypercubeSample()`.

Finally, we have the option to save the trace of each optimization run. For instance, if we want to use `Ipopt` and save the trace for each run, while ensuring reproducibility by setting a seed, we can use the following code:

```julia
using PEtab
using Ipopt

petab_model = PEtabModel(path_yaml)
petab_problem = PEtabODEProblem(petab_model)
res = calibrate_model_multistart(petab_problem, IpoptOptimiser(false), 10, dir_save,
                               save_trace=true,
                               seed=123)
print(res)
```
```
PEtabMultistartOptimisationResult
--------- Summary ---------
min(f)                = 1.38e+02
Parameters esimtated  = 9
Number of multistarts = 10
Optimiser algorithm   = Ipopt_user_Hessian
```

In this example, we use `save_trace=true` to enable trace saving and set `seed=123` for reproducibility. We can access the traces for the first run as follows:

```julia
res.runs[1].xtrace
res.runs[1].ftrace
```

## Single-Start Parameter Estimation

If we want to perform single-start parameter estimation instead of multistart, we can use the `calibrate_model` function. This function runs a single optimization from a given initial guess.

Given a starting point `p0` which can be generated by the `generate_startguesses` function, and that we want to use Ipopt for optimization the model can be parameter estimated via:

```julia
p0 = generate_startguesses(petab_problem, 1)
res = calibrate_model(petab_problem, p0, IpoptOptimiser(false),
                      options=IpoptOptions(max_iter = 1000))
print(res)
```
```
PEtabOptimisationResult
--------- Summary ---------
min(f)                = 1.38e+02
Parameters esimtated  = 9
Optimiser iterations  = 31
Run time              = 1.9e+00s
Optimiser algorithm   = Ipopt_user_Hessian
```

The results are returned as a `PEtabOptimisationResult`, which includes the following information: minimum parameter values found (`xmin`), smallest objective value (`fmin`), number of iterations, runtime, whether the optimizer converged, and optionally, the trace if `save_trace=true`.

Note that `generate_startguesses` can generate several start-guesses within the bounds of the `PEtabODEProblem` with any sampling method from [QuasiMonteCarlo](https://github.com/SciML/QuasiMonteCarlo.jl). For example, to generate 10 start guesses with Sobol sampling do:

```julia
using QuasiMonteCarlo
p0 = generate_startguesses(petab_problem, 10, 
                           sampling_method=SobolSample())
```

Additionally, if an initialization prior is specified for a parameter, the start-guesses for that parameter will be sampled from the provided prior clipped by the upper and lower bounds.