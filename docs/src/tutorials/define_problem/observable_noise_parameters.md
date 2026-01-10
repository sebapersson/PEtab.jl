# [Observable and noise parameters](@id petab_observable_options)

A `PEtabObservable` links model outputs to measured data can sometimes depend on non-model
parameters such as scaling and/or offset parameters (e.g. when measurements are
relative but the model output is absolute scale). Such parameters also sometimes vary
between measurements, for example when data were collected using different assays.

This is handled via **observable parameters** and **noise parameters**. This tutorial shows
how to define these parameters and how to optionally make them time-point specific. As a
running example, we use the Michaelis–Menten model from the
[starting tutorial](@ref tutorial).

```@example 1
using Catalyst, PEtab

rn = @reaction_network begin
    @parameters S0 c3=3.0
    @species begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.0
        P(t) = 0.0
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_S0 = PEtabParameter(:S0)
nothing # hide
```

## Specifying noise and observable parameters

Noise and/or observable parameters should first be defined as `PEtabParameter`s (can be
estimated or fixed), and then referenced in `PEtabObservable` formulas. For example, assume
the observable `P` is measured on a relative scale, requiring a scale and offset (`scale_p`,
`offset_p`), while `S + E` is measured with noise `sigma`. First, define the parameters:

```@example 1
# offset_p not estimated (fixed to 2.0)
p_scale_p  = PEtabParameter(:scale_p)
p_offset_p = PEtabParameter(:offset_p; estimate = false, value = 2.0)
p_sigma    = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_S0, p_scale_p, p_offset_p, p_sigma]
nothing # hide
```

Then define the observables using these parameters:

```@example 1
@unpack S, E, P = rn
@parameters scale_p offset_p sigma
obs1 = PEtabObservable(:petab_obs1, scale_p * P + offset_p, 1.0)
obs2 = PEtabObservable(:petab_obs2, S + E, sigma)
observables = [obs1, obs2]
```

With observables defined, a `PEtabModel` can be created as usual given a measurement table:

```@example 1
using DataFrames
measurements = DataFrame(
    obs_id      = ["petab_obs1", "petab_obs2", "petab_obs1", "petab_obs2"],
    time        = [1.0, 10.0, 1.0, 20.0],
    measurement = [0.7, 0.1, 1.0, 1.5],
)
petab_model = PEtabModel(rn, observables, measurements, pest)
petab_prob = PEtabODEProblem(petab_model)
nothing # hide
```

!!! note "When formulas use non-model quantities"
    If the observable or noise formula depends on non-model quantities (e.g. scale/offset),
    define it in `PEtabObservable` (as above) rather than inside the model system. This
    lets PEtab.jl treat such parameters as non-dynamic and compute gradients more
    efficiently.

## Time-point-specific observable and noise parameters

Time-point-specific parameters are defined by (i) using special parameter names in the
`PEtabObservable` formulas, and (ii) providing their values in the measurement table. For
instance, assume the first observable (`P`) is on a relative scale with two
time-point-specific observable parameters (scale and offset), and `:petab_obs2`
uses a time-point-specific noise parameter:

```@example 1
@unpack S, E, P = rn
@parameters observableParameter1 observableParameter2 noiseParameter1
obs1 = PEtabObservable(:petab_obs1, observableParameter1 * P + observableParameter2, 1.0)
obs2 = PEtabObservable(:petab_obs2, S + E, noiseParameter1)
observables = [obs1, obs2]
nothing # hide
```

!!! note "Required naming convention"
    Time-point-specific parameters must be named `observableParameter{n}` and
    `noiseParameter{n}`, with `n` starting from 1.

Values for time-point-specific parameters are provided per measurement row via the optional
columns `observable_parameters` and `noise_parameters` (column names matter, but not order).
Values are given as a semicolon-separated list matching the parameter index `n`:

| obs_id (str) | time (float) | measurement (float) | observable_parameters (str \| float) | noise_parameters (str \| float) |
| ------------ | ------------ | ------------------- | ------------------------------------ | ------------------------------- |
| petab_obs1   | 1.0          | 0.7                 | 3.0;4.0                              | sigma                           |
| petab_obs2   | 10.0         | 0.1                 |                                      | 1.0                             |
| petab_obs1   | 1.0          | 1.0                 | 2.0;3.0                              | sigma                           |
| petab_obs2   | 20.0         | 1.5                 |                                      | 0.5                             |

Note, for the `observable_parameters` and `noise_parameters` columns:

- Leave the column empty (`missing`) if an observable has no parameters of that type.
- Multiple values are separated by `;` (e.g. `"3.0;4.0"`).
- `PEtabParameter` ids are allowed (e.g. `"sigma"`), and mixed entries like `"sigma;1.0"`
    are also allowed.
- If an observable uses time-point-specific parameters, values must be provided for each
  measurement of that observable.

With a measurement table in the correct format, a `PEtabModel` can be created as usual:

```@example 1
using DataFrames
measurements = DataFrame(
    obs_id = ["petab_obs1", "petab_obs2", "petab_obs1", "petab_obs2"],
    time = [1.0, 10.0, 1.0, 20.0],
    measurement = [0.7, 0.1, 1.0, 1.5],
    observable_parameters = ["3.0;4.0", missing, "2.0;3.0", missing],
    noise_parameters = ["sigma", "1.0", "sigma", "0.5"],
)

petab_model = PEtabModel(rn, observables, measurements, pest)
petab_prob = PEtabODEProblem(petab_model)
```
