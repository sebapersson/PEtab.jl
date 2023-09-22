# [Available Optimizers](@id options_optimizers)

PEtab.jl offers an interface to several popular optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides) for performing parameter estimation. Below, you find the available options for each optimizer.

## Optim

PEtab.jl supports three optimization methods from [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/): LBFGS, BFGS, and IPNewton (Interior-point Newton). You can further customize the optimization by providing options via `Optim.Options()`. A complete list of available options can be found [here](https://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/).

For example, LBFGS with 10,000 iterations can be used via:

```julia
res = calibrate_model(petab_problem, p0, Optim.LBFGS(),
                     options=Optim.Options(iterations = 10000))
```

If no options are provided, the default values are used:

```julia
Optim.Options(iterations = 1000,
              show_trace = false,
              allow_f_increases = true,
              successive_f_tol = 3,
              f_tol = 1e-8,
              g_tol = 1e-6,
              x_tol = 0.0)
```

## Ipopt

[Ipopt](https://coin-or.github.io/Ipopt/) is an Interior-point Newton method designed for nonlinear optimization. In PEtab.jl, you can configure Ipopt to use either the Hessian method from the `PEtabODEProblem` or a LBFGS Hessian approximation.

To use the LBFGS Hessian approximation with Ipopt write:

```julia
using Optim
res = calibrate_model(petab_problem, p0, IpoptOptimiser(true))
```

With `true` indicates the use of the LBFGS approximation.

To use the method in the `PEtabODEProblem`, and want to run Ipopt for 200 iterations, write:

```julia
using Ipopt
res = calibrate_model(petab_problem, p0, IpoptOptimiser(false),
                     options=IpoptOptions(max_iter = 200))
```

In this case, `false` means you are using the method defined in the `PEtabODEProblem`, and the `max_iter` option limits the optimization to 200 iterations.

You can further configure Ipopt's behavior using `IpoptOptions`. Available options are:

- `print_level`: Output verbosity level (valid values: 0 ≤ print_level ≤ 12)
- `max_iter`: Maximum number of iterations
- `tol`: Relative convergence tolerance
- `acceptable_tol`: Acceptable relative convergence tolerance
- `max_wall_time`: Maximum wall time for optimization
- `acceptable_obj_change_tol`: Stopping criterion based on objective function change.

If no options are provided, PEtab.jl defaults to:

```julia
using Ipopt
IpoptOptions(;print_level::Int64=0,
             max_iter::Int64=1000,
             tol::Float64=1e-8,
             acceptable_tol::Float64=1e-6,
             max_wall_time::Float64=1e20,
             acceptable_obj_change_tol::Float64=1e20)
```

## Fides

[Fides.py](https://github.com/fides-dev/fides) is a trust-region Newton method known for box-constrained optimisation problems. It is particularly efficient when computing the full Hessian is computationally expensive, but a Gauss-Newton Hessian approximation is feasible.

In PEtab.jl, you can use Fides for parameter estimation, but note that Fides is a Python library. To set up the necessary environment for Fides, make sure you have [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) installed in your Julia environment. You will also need to build PyCall with a Python environment that has Fides installed:

```julia
using PyCall

# Set the path to your Python executable
path_python_exe = "path_python"

# Set the PYTHON environment variable to the path of your Python executable
ENV["PYTHON"] = path_python_exe

# Build PyCall with the specified Python environment
import Pkg
Pkg.build("PyCall")
```

!!!note
    `path_python_exe` should point to your Python executable, and it depends on the system configuration

Fides can be configured to use different Hessian methods; the method from the `PEtabODEProblem` or various approximation methods:

- `:BB`: Broyden's "bad" method
- `:BFGS`: Broyden-Fletcher-Goldfarb-Shanno update strategy
- `:BG`: Broyden's "good" method
- `:Broyden`: BroydenClass Update scheme
- `:SR1`: Symmetric Rank 1 update
- `:SSM`: Structured Secant Method
- `:TSSM`: Totally Structured Secant Method

To use Fides with a specific Hessian approximation method, such as BFGS, write:

```julia
using PyCall
res = calibrate_model(petab_problem, p0, Fides(:BFGS))
```

If you prefer to use the Hessian method from the `PEtabODEProblem` and limit Fides to 200 iterations, write:

```julia
using PyCall
res = calibrate_model(petab_problem, p0, Fides(nothing),
                     options=py"{'maxiter' : 1000}"o)
```

Fides options are specified using a Python dictionary. Available options and their default values can be found in the Fides [documentation](https://fides-optimizer.readthedocs.io/en/latest/generated/fides.constants.html).