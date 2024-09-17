"""
    petab_select(path_yaml, alg; kwargs...)

Given a PEtab-select YAML file perform model selection with the algorithms specified in the
YAML file using a provided optmization algorithm `alg`.

Results are written to a YAML file in the same directory as the PEtab-select YAML file.

Each candidate model produced during the model selection undergoes parameter estimation
using local multi-start optimization. Three alg are supported: `optimizer=Fides()`
(Fides Newton-trust region), `optimizer=IPNewton()` from Optim.jl, and `optimizer=LBFGS()`
from Optim.jl. Additional keywords for the optimisation are `nmultistarts::Int`- number
of multi-starts for parameter estimation (defaults to 100) and
`optimizationSamplingMethod` - which is any sampling method from QuasiMonteCarlo.jl for
generating start guesses (defaults to LatinHypercubeSample).

Simulation options can be set using any keyword argument accepted by the `PEtabODEProblem` function.
For example, setting `gradient_method=:ForwardDiff` specifies the use of forward-mode automatic differentiation for
gradient computation. If left blank, we automatically select appropriate options based on the size of the problem.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you
    must load Ipopt with `using Ipopt`. To use Fides, load PyCall with `using PyCall`
    and ensure Fides is installed (see documentation for setup).
"""
function petab_select end
