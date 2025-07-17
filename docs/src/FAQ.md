# [Frequently Asked Questions](@id FAQ)

## [Why do I encounter installation problems?](@id install_fail)

PEtab.jl is regularly tested on, and should be installable on Linux, macOS and Windows. If you encounter installation issues on these systems, we recommend checking the following two common causes:

1. an incorrectly installed or outdated Julia version,  
2. outdated package dependencies.

First, ensure that a supported Julia version is used. PEtab.jl is tested with Julia LTS version **1.10** and the latest stable version. Using an earlier version may result in installation failures. To reliably install and manage Julia versions across operating systems, we strongly recommend using [juliaup](https://github.com/JuliaLang/juliaup). If you are constrained to using an older Julia version, for example on an HPC cluster, and encounter problems, please file an issue on [GitHub](https://github.com/sebapersson/PEtab.jl/issues).

Second, installation failures may result from outdated versions of PEtab.jl dependencies. For example, if PEtab.jl is installed into the global Julia environment, older versions of other packages may prevent the latest version from being installed. This can cause installation failures or break tutorials and example code. To avoid this, it is recommended to install PEtab.jl in a new, isolated environment. For example, to install it in an environment named `petab_project`, run the following in the Julia REPL:

```julia
using Pkg
Pkg.activate("petab_project")
Pkg.add("PEtab")
# Add any additional packages as needed
```

If you need to install PEtab.jl into an existing environment and encounter issues, updating all packages may resolve the problem:

```julia
Pkg.update()
```

This is because PEtab.jl depends on numerous packages from the actively developing Julia [SciML ecosystem](https://sciml.ai/). New releases of these dependencies sometimes introduce breaking changes that are not always caught by test suites (e.g., see [this issue](https://github.com/SciML/Catalyst.jl/issues/1075)). In other words, PEtab.jl is not compatible with all versions of packages like Catalyst.jl, which can cause issues if an incompatible version is already installed in the environment.

Lastly, if you have tried everything above and still experience installation issues, it is likely a bug in PEtab. In this case, please open an issue on [GitHub](https://github.com/sebapersson/PEtab.jl).

## How do I check that I implemented my parameter estimation problem correctly?

After creating a `PEtabODEProblem`, it is important to check that everything works as expected. Since PEtab.jl creates parameter estimation problems, this means checking that the objective function (the problem likelihood) is computable, because if not, running parameter estimation will only return `NaN` or `Inf`.

The first step to verify that the likelihood is computable is to check if the objective function can be computed for the nominal parameters:

```julia
x = get_x(petab_prob)
petab_prob.nllh(x)
```

The nominal values can be specified when creating a `PEtabParameter` or in the parameters table if the problem is provided in the PEtab standard format (otherwise, they default to the mean of the parameter bounds). If the problem is correctly specified, the likelihood should be computable for these values. However, sometimes the nominal values can be poor choices (far from the 'true' parameters as we do not know them, hence the need for PEtab.jl), and the code above may return `Inf` because the ODE cannot be solved. If this happens, check if the likelihood can be computed for random parameters:

```julia
get_startguesses(petab_prob, 10)
```

Specially, the [`get_startguesses`](@ref) function tries to find random parameter vectors for which the likelihood can be computed. If this function fails to return a parameter set, there is likely an issue with the problem formulation.

If the objective function cannot be computed, check out the tips below. If none of the suggestions help, please file an [issue](https://github.com/sebapersson/PEtab.jl/issues) on GitHub.

## Why do I get `NaN` or `Inf` when computing the objective function or during parameter estimation?

Sometimes, when computing the likelihood (`petab_prob.nllh(x)`) or during parameter estimation, `Inf` or `NaN` may be returned. This can be due to several reasons.

`Inf` is returned when the ODE model cannot be solved. When this happens, a warning like `Failed to solve ODE model` should be printed. If no ODE solver warning is shown, check that the observable formulas and noise formulas cannot evaluate to `Inf` (e.g., there are no expressions that can evaluate to `log(0)`). If neither of these reasons causes `Inf` to be returned, please file an [issue](https://github.com/sebapersson/PEtab.jl/issues) on GitHub. For how to deal with ODE solver warnings, see one of the questions below.

If `NaN` is returned, the model formulas are likely ill-formulated. In PEtab.jl, the most common cause of `NaN` being returned is that `log` is applied to a negative value, often due to an ill-formulated noise formula. For example, consider the observable `h = PEtabObservable(X, sigma * X)`, where `X` is a model species and `sigma` is a parameter. When computing the objective value (likelihood) for this observable, the `log` of the noise formula `sigma * X` is evaluated. Even if the model uses mass-action kinetics and `X` should never go below zero, in practice, numerical noise during ODE solving can cause `X` to become negative, leading to a negative argument for `log`. Therefore, a more stable noise formula than the one above would be `sigma1 + sigma2 * X`.

## Why do I get the error *MethodError: Cannot convert an object of type Nothing to an object of type Real*?

This error likely occurs because some model parameters have not been defined. For example, consider the observable `h = PEtabObservable(X, sigma)`, where `X` is a model species and `sigma` is a parameter. If `sigma` has not been defined as a `PEtabParameter`, the above error will be thrown when computing the objective function. This also applies to misspellings. For example, if the observable is defined as `h = PEtabObservable(X, sigma1)` but only `sigma` (not `sigma1`) is defined as a `PEtabParameter`, the same error will be thrown.

## Why are my parameter values (e.g., start-guesses) negative?

When creating start-guesses for parameter estimation with [`get_x`](@ref) or [`get_startguesses`](@ref), the values in the returned vector(s) can sometimes be negative. As discussed in the [starting tutorial](@ref tutorial), this is because parameters are estimated on the `log10` scale by default, as this often improves performance. Consequently, when setting new parameter values manually, they should be provided on the parameter scale. It is also possible to change the parameter scale; see [`PEtabParameter`](@ref).

## I get ODE-solver warnings during parameter estimation, is my model wrong?

When doing parameter estimation, it is not uncommon for the warning `Failed to solve ODE model` to be thrown a few times. This is because when estimating model parameters with a numerical optimization algorithm that starts from random points in parameter space (e.g., when using [`calibrate_multistart`](@ref)), poor parameter values that cause difficult dynamics to solve are sometimes generated. However, if the `Failed to solve ODE model` warning is thrown frequently, there is likely a problem with the model structure, or a suboptimal ODE solver is used. We recommend first checking if the issue is related to the ODE solver.

A great collection of tips for dealing with different ODE solver warnings can be found [here](https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#faq). Briefly, it can be helpful to adjust the tolerances in [`ODESolver`](@ref), as the default settings are quite strict. Further, if `maxiters` warnings are thrown, increasing the number of maximum iterations might help. Lastly, it can also be worthwhile to try different ODE solvers. Even though the [default solver](@ref default_options) often performs well, every problem is unique. For hard-to-solve models, it can therefore be useful to try solvers like `Rodas5P`, `QNDF`, `TRBDF2`, or `KenCarp4`.

If changing ODE solver settings does not help, something may be wrong with the model structure (e.g., the model may not be coded correctly). However, it should also be kept in mind that some models are just hard to solve/simulate. Therefore, even if many warnings are thrown, a multi-start parameter estimation approach can still sometimes find a set of parameters that fits the data well.

## How do I turn off ODE solver printing?

When performing parameter estimation, as discussed above warnings are thrown when the ODE solver fails to solve the underlying ODE model. By default, we do not disable ODE solver warnings, as it can be beneficial to see them. In particular, if warnings are thrown frequently, it may indicate that something is wrong with the model structure (e.g. the model was not coded correctly) or that a sub-optimal (e.g., non-stiff) ODE solver was chosen when a stiff one should be used. Regardless, when running parameter estimation, it might be preferable not to have the terminal cluttered with warnings. You can turn off the warnings by setting the `verbose = false` option in the `ODESolver`:

```julia
osolver = ODESolver(Rodas5P(); verbose = false)
petab_prob = PEtabODEProblem(model; odesolver = osolver)
```

For which ODE solver to choose when manually setting the `ODESolver`, see [this](@ref default_options) page.

## How do I create a parameter estimation problem for an SBML model?

If your model is in the [SBML](https://sbml.org/) standard format, there are two ways to create a parameter estimation problem:

1. Formulate the problem in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format (recommended). PEtab is a standard format for parameter estimation that assumes the model is in the SBML format. We recommend creating problems in this format, as it allows for the exchange of parameter estimation workflows and is more reproducible. A guide on how to create problems in this standard format can be found [here](https://petab.readthedocs.io/en/latest/), and a tutorial on importing problems can be found [here](@ref import_petab_problem).

2. Import the model as a [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem` with [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl) (see the SBMLImporter [documentation](https://sebapersson.github.io/SBMLImporter.jl/stable/) for details). As demonstrated in the starting [tutorial](@ref tutorial), a `ReactionSystem` is one of the model formats that PEtab.jl accepts for creating a parameter estimation problem directly in Julia.
