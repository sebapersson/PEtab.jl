# [Noise and Observable Parameters](@id time_point_parameters)

It is common to measure an observable, such as a product `P`, with different experimental assays. This can lead to variation in the measurement noise parameter `sigma` between measurements for the same observable, and additionally, different assays might measure `P` on different relative scales, necessitating different scale and offset parameters. Measurement-specific parameters can be handled by defining noise and observable parameters. To demonstrate how, let us continue with the same enzyme kinetics model as in the [Creating a PEtab Parameter Estimation Problem in Julia](@ref define_in_julia) tutorial.

```julia
using Catalyst 
using Distributions
using PEtab

rn = @reaction_network begin
    @parameters se0
    @species SE(t) = se0  # se0 = initial value for S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
state_map =  [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]

_c3 = PEtabParameter(:c3, scale=:log10)
_se0 = PEtabParameter(:c3, prior=LogNormal(1.0, 0.5), prior_on_linear_scale=true)
_c2 = PEtabParameter(:c2)
_sigma = PEtabParameter(:sigma)
_scale = PEtabParameter(:scale)
_offset = PEtabParameter(:offset)
parameters = [_c2, _c3, _se0, _sigma, _scale, _offset]

condition_c0 = Dict(:S => 5.0)
condition_c1 = Dict(:S => 2.0)
simulation_conditions = Dict("c0" => condition_c0, 
                             "c1" => condition_c1)
```

Now, to incorporate time-point-specific noise and measurement parameters into the observable formula, we encode them in the form `observableParameter...` and `noiseParameters`.

```julia
@unpack P, E, SE = rn
@parameters noiseParameter1_obs_P observableParameter1_obs_P observableParameter2_obs_P
obs_P = PEtabObservable(observableParameter1_obs_P * P + observableParameter2_obs_P, noiseParameter1_obs_P * P)
obs_Sum = PEtabObservable(E + SE, 3.0, transformation=:log)
observables = Dict("obs_P" => obs_P, 
                   "obs_Sum" => obs_Sum) 
```

!!! note
    noiseParameters and observableParameter must always be on the format `observableParameter${n}_${observableId}` and `noiseParameter${n}_${observableId}`with n starting from 1, in order to correctly map parameters when building the objective function.

Next, we provide values for these parameters in the measurement data. These parameters can either be specified as values or parameters that have been defined as a `PEtabParameter`. In the case of multiple noise or observable parameters for a measurement, they are delimited by a semicolon:

| simulation_id (str) | obs_id (str) | time (float) | measurement (float) | observable_parameters (str\|float) | noise_parameters (str\|float) |
|---------------------|--------------|--------------|---------------------|------------------------------------|-------------------------------|
| c0                  | obs_P        | 0.0          | 0.7                 |                                    |                               |
| c0                  | obs_Sum      | 10.0         | 0.1                 | scale;offset               | sigma                         |
| c1                  | obs_P        | 1.0          | 1.0                 |                                    |                               |
| c1                  | obs_Sum      | 20.0         | 1.5                 | 1.0;1.0                            | 4.0                           |

Note, in case an observable (like `obs_P`) does not have noise-or observable-parameters we do not have to provide any values for it. For the first observation for observable `obs_Sum` the parameter `scale` maps to `observableParameter1_obs_Sum` while offset maps to `observableParameter2_obs_Sum`. In Julia, the measurement data would look like:

```julia
measurements = DataFrame(
    simulation_id=["c0", "c0", "c0", "c0"],
    obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
    time=[0.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5], 
    observable_parameters=[missing, "scale;offset", missing, "1.0;1.0"],
    noise_parameters=[missing, "sigma", missing, 4.0]
)
```

With this setup, you can create a `PEtabODEProblem` for model calibration:

```julia
petab_model = readPEtabModel(
    rn, simulation_conditions, observables, measurements,
    parameters, stateMap=state_map, parameterMap=parameter_map, verbose=true
)
petab_problem = createPEtabODEProblem(petab_model)  
```