# Optimization (parameter estimation)

PEtab.jl can easily integrate with various optimization packages like [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides), as demonstrated in our [examples](https://github.com/sebapersson/PEtab.jl/tree/main/examples).

Based on our extensive benchmarking, here's a good rule of thumb for selecting an optimizer:

- If you're able to provide a full Hessian, then the Interior-point Newton method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) generally performs better than the trust-region method in Fides.py.
- If you can only provide a Gauss-Newton Hessian approximation (not the full Hessian), then the Newton trust-region method in Fides.py usually outperforms the interior-point method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

!!! note
    Each problem is distinct, and although the suggested option is typically effective, it may not be the ideal choice for a particular model.
