# Tutorials

The PEtab.jl tutorials cover how to set up parameter-estimation problems with various
features, and how to run and configure parameter estimation.

## Creating a parameter estimation problem

- [Simulation conditions](@ref petab_sim_cond): Measurements collected under different
  experimental conditions (e.g. simulations use different initial values).
- [Pre-equilibration](@ref define_with_ss): Enforce a steady state before the model is
  matched against data (pre-equilibration).
- [Simulation condition-specific parameters](@ref condition_parameters): Subset of model
  parameters which are estimated take different across simulation conditions.
- [Observable and noise parameters](@ref petab_observable_options): Observable/noise
  parameters in `PEtabObservable` formulas that are not part of the model system (e.g. scale/offset), optionally time-point-specific.
- [Events/callbacks](@ref define_events): Time- or state-triggered events/callbacks.
- [Import PEtab standard format](@ref import_petab_problem): Load problems from PEtab
  standard format.
- Model definition: More on defining `ReactionSystem` and `ODESystem` models can be
  found in the [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) and
  [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/dev/) documentation
  respectively.

## Parameter estimation

- [Parameter estimation extended tutorial](@ref pest_methods): Extended tutorial on
  estimation functionality (e.g. multi-start, Optimization.jl integration).
- [Available optimization algorithms](@ref options_optimizers): Supported and recommended
  algorithms.
- [Plotting parameter estimation results](@ref pest_plotting): Plotting options for
  parameter estimation output.
- [Model selection](@ref petab_select): Automatic model selection with PEtab-Select.
- [Wrapping optimization packages](@ref wrap_est): How to use a `PEtabODEProblem` directly
  with an optimization package such as Optim.jl.
- [Bayesian inference](@ref bayesian_inference): Sampling-based inference (e.g. NUTS and
  AdaptiveMCMC).
