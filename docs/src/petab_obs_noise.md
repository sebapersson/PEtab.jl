# [Noise and Observable Parameters](@id time_point_parameters)

Sometimes a model observable (e.g., a protein) is measured using different experimental assays. This can result in measurement noise parameter `σ` being different between measurements for the same observable. Additionally, if the observable is measured on a relative scale, the observable offset and scale parameters that link the model output scale to the measurement data scale might differ between measurements. From a modeling viewpoint, this can be handled by introducing time-point-specific noise and/or observable parameters.

This tutorial covers how to specify time-point specific observable and noise parameters for a `PEtabModel`. As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial](@ref tutorial). Even though the code below encodes the model as a `ReactionSystem`, everything works exactly the same if the model is encoded as an `ODESystem`.

```@example 1
using Catalyst, PEtab

rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]

p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]
nothing # hide
```

## Specifying Noise and Observable Parameters

Time-point-specific parameters are handled by encoding observable and noise parameters in the `PEtabObservable`, followed by setting values for these parameters in the measurements `DataFrame`. For instance, assume that the measurement noise is time-point specific for the observable `obs_sum`. Then, the first step is to add a noise parameter of the form `noiseParameter...` in the `PEtabObservable`:

```@example 1
@unpack E, S = rn
@parameters noiseParameter1_obs_sum
obs_sum = PEtabObservable(S + E, noiseParameter1_obs_sum)
```

Additionally, assume that data is measured on a relative scale, where the scale and offset parameters vary between time points for the observable `obs_p`. Then, the first step for this observable is to add observable parameters of the form `observableParameter...`:

```@example 1
@unpack P = rn
@parameters observableParameter1_obs_p observableParameter2_obs_p
obs_p = PEtabObservable(observableParameter1_obs_p * P + observableParameter2_obs_p, 3.0)
```

Given this, the observables are collected in a `Dict` as usual:

```@example 1
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)
nothing # hide
```

!!! note
    Noise and observable parameters must follow the format `observableParameter${n}_${observableId}` and `noiseParameter${n}_${observableId}`, with `n` starting from 1, to ensure correct parameter mapping when building the PEtab problem. This follows the PEtab specification, and more details can be found [here](https://petab.readthedocs.io/en/latest/index.html).

## Mapping Measurements to Time-Point Specific Parameters  

To link the measurements to time-point-specific noise and/or observable parameters, values for these parameters must be specified in the measurements `DataFrame`. These values can be either constant numerical values or any defined `PEtabParameter`. For our working example, a valid measurement table would look like this (the column names matter, but not the order):

| obs_id (str) | time (float) | measurement (float) | observable_parameters (str \| float) | noise_parameters (str \| float) |
|--------------|--------------|---------------------|--------------------------------------|---------------------------------|
| obs_p        | 1.0          | 0.7                 |                                      | sigma                           |
| obs_sum      | 10.0         | 0.1                 | 3.0; 4.0                             |                                 |
| obs_p        | 1.0          | 1.0                 |                                      | sigma                           |
| obs_sum      | 20.0         | 1.5                 | 2.0; 3.0                             |                                 |

In particular, the follow consideration apply to the measurements table:

- If an observable does not have noise or observable parameters (e.g., `obs_p` above lacks observable parameters), the corresponding column should be left empty.
- For multiple parameters, values are separated by a semicolon (e.g., for `obs_sum`, we have `3.0; 4.0`).
- The values for noise and observable parameters can be either numerical values or any defined `PEtabParameter` (e.g., `sigma` above). Combinations are also allowed, so `sigma;1.0` is valid.
- If an observable has noise and/or observable parameters, values for these must be specified for each measurement of that observable.

For our working example, the measurement data would in Julia look like:

```@example 1
using DataFrames
measurements = DataFrame(
    obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5], 
    observable_parameters=[missing, "3.0;4.0", missing, "2.0;3.0"],
    noise_parameters=["sigma", missing, "sigma", missing]
)
```

## Bringing It All Together

Given `observables` and `measurements` in the correct format, it is straightforward to create a PEtab problem with time-point-specific parameters by creating the `PEtabModel` as usual:

```@example 1
model = PEtabModel(rn, observables, measurements, pest; speciemap = speciemap)
petab_prob = PEtabODEProblem(model)
```
