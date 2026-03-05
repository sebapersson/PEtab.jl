# [Speeding up pre-simulation SciML models](@id performance_pre_simulate)

In the pre-simulation scenario, the ML model is evaluated once per simulation condition
before model simulation. This allow performance optimizations, as ML gradients can be
computed in three steps: (1) compute the Jacobian of the ML model output with respect to its
parameters; (2) compute the gradient of the objective with respect to the ODE parameters
(including those set by the ML model); and (3) obtain the ML-parameter gradient via a
Jacobian-vector product between the Jacobian from (1) and the gradient from (2).

This behavior is controlled by `split_over_conditions` when constructing `PEtabODEProblem`.
Currently, `split_over_conditions = true` is the default and enables precompilation of the
reverse pass. To see the effect, lets compare gradient runtimes for an example where a small
convolutational neural network is used to set parameters in the PEtab problem:

```@example 1
using Catalyst, DataFrames, Lux, PEtab, StableRNGs

sys = @reaction_network begin
    @parameters S0
    @species begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.1
        P(t) = 0.1
    end
    @observables begin
        obs1 ~ S + E
        obs2 ~ P
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

lux_model = Lux.Chain(
    Conv((5, 5), 3 => 1),
    FlattenLayer(),
    Dense(36 => 1, Lux.softplus),
)
ml_model = MLModel(
    :net1, lux_model, true; inputs = [:input1], outputs = [:c3]
)

# Simulation condition specific input
rng = StableRNG(1)
input_data1 = rand(rng, 10, 10, 3, 1)
input_data2 = rand(rng, 10, 10, 3, 1)
simulation_conditions = [
    PEtabCondition(:cond1, :input1 => input_data1),
    PEtabCondition(:cond2, :input1 => input_data2),
]

# Observables
@parameters sigma
observables = [
    PEtabObservable(:obs_p, :obs1, 3.0),
    PEtabObservable(:obs_sum, :obs2, sigma),
]

# Parameter to estimate
pest = [
    PEtabParameter(:c1),
    PEtabParameter(:c2),
    PEtabParameter(:S0),
    PEtabParameter(:sigma),
    PEtabMLParameter(:net1), # ML parameters
]

# Measurement data
measurements = DataFrame(
    simulation_id = ["cond1", "cond1", "cond2", "cond2"],
    obs_id        = ["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time          = [5.0, 10.0, 1.0, 10.0],
    measurement   = [0.7, 0.1, 1.0, 1.5],
)

petab_model = PEtabModel(
    sys, observables, measurements, pest; ml_models = ml_model,
    simulation_conditions = simulation_conditions,
)
nothing # hide
```

Providing `split_over_conditions = true` noticeably reduces runtime in this example. The
effect should also be even more noticable for larger ML models:

```@example 1
using Printf

petab_prob_split = PEtabODEProblem(
    petab_model; split_over_conditions = true # default
)
petab_prob_joint = PEtabODEProblem(
    petab_model; split_over_conditions = false
)

x = get_x(petab_prob_split)
import Suppressor # hide
Suppressor.@suppress begin # hide
    _ = petab_prob_split.grad(x) # hide
    _ = petab_prob_joint.grad(x) # hide
end # hide
t_split = @elapsed petab_prob_split.grad(x)
t_joint = @elapsed petab_prob_joint.grad(x)
@printf("Wall time split_over_conditions=true : %.2e seconds\n", t_split)
@printf("Wall time split_over_conditions=false: %.2e seconds\n", t_joint)
```
