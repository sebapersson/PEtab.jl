# Optimization (parameter estimation)

PEtab.jl is written to easily integrate with available optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [Fides.py](https://github.com/fides-dev/fides). In the [examples](https://github.com/sebapersson/PEtab.jl/tree/main/examples) we show how to use these together with PEtab.jl.

Based on an extensive benchmark a good rule of thumb when choosing optimizer is:

* If you can provide a full Hessian the Interior-point Newton method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) often outperforms the trust-region method in Fides.py.
* In case you cannot provide the full Hessian but the Gauss-Newton hessian approximation, the Newton trust-region method in Fides.py often outperforms the interior-point method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

!!! note
    Every problem is unique, and the recommended choice here will often work well but might not be optimal for a specific model
