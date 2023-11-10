"""
    calibrate_model(petab_problem::PEtabODEProblem,
                    p0::Vector{Float64},
                    alg;
                    save_trace::Bool=false,
                    options=algOptions)::PEtabOptimisationResult

Parameter estimate a model for a PEtabODEProblem using an optimization algorithm `alg` and an initial guess `p0`.

The optimization algorithm `alg` can be one of the following:
- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) LBFGS, BFGS, or IPNewton methods
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/) interior-point optimizer
- [Fides](https://github.com/fides-dev/fides) Newton trust region method

Each algorithm accepts specific optimizer options in the format of the respective package. For a
comprehensive list of available options, please refer to the main documentation.

If you want the optimizer to return parameter and objective trace information, set `save_trace=true`.
Results are returned as a `PEtabOptimisationResult`, which includes the following information: minimum
parameter values found (`xmin`), smallest objective value (`fmin`), number of iterations, runtime, whether
the optimizer converged, and optionally, the trace.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`.
    To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).

## Examples
```julia
# Perform parameter estimation using Optim's IPNewton with a given initial guess
using Optim
res = calibrate_model(petab_problem, p0, Optim.IPNewton();
                     options=Optim.Options(iterations = 1000))
```
```julia
# Perform parameter estimation using Fides with a given initial guess
using PyCall
res = calibrate_model(petab_problem, p0, Fides(nothing);
                     options=py"{'maxiter' : 1000}"o)
```
```julia
# Perform parameter estimation using Ipopt and save the trace
using Ipopt
res = calibrate_model(petab_problem, p0, IpoptOptimiser(false);
                     options=IpoptOptions(max_iter = 1000),
                     save_trace=true)
```
"""
function calibrate_model end


"""
    calibrate_model_multistart(petab_problem::PEtabODEProblem,
                             alg,
                             n_multistarts::Signed,
                             dir_save::Union{Nothing, String};
                             sampling_method=QuasiMonteCarlo.LatinHypercubeSample(),
                             options=algOptions,
                             seed=nothing,
                             save_trace::Bool=false)::PEtabMultistartOptimisationResult

Perform multistart optimization for a PEtabODEProblem using the algorithm `alg`.

The optimization algorithm `alg` can be one of the following:
- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) LBFGS, BFGS, or IPNewton methods
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/) interior-point optimizer
- [Fides](https://github.com/fides-dev/fides) Newton trust region method

For each algorithm, optimizer options can be provided in the format of the respective package.
For a comprehensive list of available options, please refer to the main documentation. If you want the optimizer
to return parameter and objective trace information, set `save_trace=true`.

Multistart optimization involves generating multiple starting points for optimization runs. These starting points
are generated using the specified `sampling_method` from [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl),
with the default being LatinHypercubeSample, a method that typically produces better results than random sampling.
For reproducibility, you can set a random number generator seed using the `seed` parameter.

If `dir_save` is provided as `nothing`, results are not written to disk. Otherwise, if a directory path is provided,
results are written to disk. Writing results to disk is recommended in case the optimization process is terminated
after a number of optimization runs.

The results are returned as a `PEtabMultistartOptimisationResult`, which stores the best-found minima (`xmin`),
smallest objective value (`fmin`), as well as optimization results for each run.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`.
    To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).

## Examples
```julia
# Perform 100 optimization runs using Optim's IPNewton, save results in dir_save
using Optim
dir_save = joinpath(@__DIR__, "Results")
res = calibrate_model_multistart(petab_problem, Optim.IPNewton(), 100, dir_save;
                               options=Optim.Options(iterations = 1000))
```
```julia
# Perform 100 optimization runs using Fides, save results in dir_save
using PyCall
dir_save = joinpath(@__DIR__, "Results")
res = calibrate_model_multistart(petab_problem, Fides(nothing), 100, dir_save;
                               options=py"{'maxiter' : 1000}"o)
```
```julia
# Perform 100 optimization runs using Ipopt, save results in dir_save. For each
# run save the trace
using Ipopt
dir_save = joinpath(@__DIR__, "Results")
res = calibrate_model_multistart(petab_problem, IpoptOptimiser(false), 100, dir_save;
                               options=IpoptOptions(max_iter = 1000),
                               save_trace=true)
```
"""
function calibrate_model_multistart end


"""
    run_PEtab_select(path_yaml, alg; <keyword arguments>)

Given a PEtab-select YAML file perform model selection with the algorithms specified in the YAML file.

Results are written to a YAML file in the same directory as the PEtab-select YAML file.

Each candidate model produced during the model selection undergoes parameter estimation using local multi-start
optimization. Three alg are supported: `optimizer=Fides()` (Fides Newton-trust region), `optimizer=IPNewton()`
from Optim.jl, and `optimizer=LBFGS()` from Optim.jl. Additional keywords for the optimisation are
`n_multistarts::Int`- number of multi-starts for parameter estimation (defaults to 100) and
`optimizationSamplingMethod` - which is any sampling method from QuasiMonteCarlo.jl for generating start guesses
(defaults to LatinHypercubeSample).

Simulation options can be set using any keyword argument accepted by the `PEtabODEProblem` function.
For example, setting `gradient_method=:ForwardDiff` specifies the use of forward-mode automatic differentiation for
gradient computation. If left blank, we automatically select appropriate options based on the size of the problem.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`. To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).
"""
function run_PEtab_select end


"""
    generate_startguesses(petab_problem::PEtabODEProblem,
                          n_multistarts::Int64;
                          sampling_method::T=QuasiMonteCarlo.LatinHypercubeSample(),
                          allow_inf_for_startguess::Bool=false,
                          verbose::Bool=false)::Array{Float64} where T <: QuasiMonteCarlo.SamplingAlgorithm

Generate `n_multistarts` initial parameter guesses within the parameter bounds in the `petab_problem` with `sampling_method`

Any sampling algorithm from QuasiMonteCarlo is supported, but `LatinHypercubeSample` is recomended as it usually
performs well.

If `n_multistarts` is set to 1, a single random vector within the parameter bounds is returned. For
`n_multistarts > 1`, a matrix is returned, with each column representing a different initial guess.

By default `allow_inf_startguess=false` - only initial guesses that result in finite cost evaluations are returned.
If `allow_inf_startguess=true`, initial guesses that result in `Inf` are allowed.

## Example
```julia
# Generate a single initial guess within the parameter bounds
start_guess = generate_startguesses(petab_problem, 1)
```

```julia
# Generate 10 initial guesses using Sobol sampling
start_guess = generate_startguesses(petab_problem, 10,
                                    sampling_method=QuasiMonteCarlo.SobolSample())
```
"""
function generate_startguesses(petab_problem::PEtabODEProblem,
                               n_multistarts::Int64;
                               sampling_method::T=QuasiMonteCarlo.LatinHypercubeSample(),
                               allow_inf_for_startguess::Bool=false,
                               verbose::Bool=false)::Array{Float64} where T <: QuasiMonteCarlo.SamplingAlgorithm

    verbose == true && @info "Generating start-guesses"
    @unpack prior_info, θ_names, lower_bounds, upper_bounds, = petab_problem

    # Nothing prevents the user from sending in a parameter vector with zero parameters
    if length(lower_bounds) == 0
        return Vector{Float64}(undef, 0)
    end

    if n_multistarts == 1
        while true
            _p::Vector{Float64} = [rand() * (upper_bounds[j] - lower_bounds[j]) + lower_bounds[j] for j in eachindex(lower_bounds)]
            # Account for potential initalisation priors
            for (θ_name, _dist) in prior_info.initialisation_distribution
                _i = findfirst(x -> x == θ_name, θ_names)
                _lb, _ub = get_bounds_prior(θ_name, petab_problem)
                _prior_samples = sample_from_prior(1, _dist, _lb, _ub)[1]
                transform_prior_samples!(_prior_samples, θ_name, petab_problem)
                _p[_i] = _prior_samples[1]
            end

            _cost = petab_problem.compute_cost(_p)
            if allow_inf_for_startguess == true
                return _p
            elseif !isinf(_cost)
                return _p
            end
        end
    end

    startguesses = Matrix{Float64}(undef, length(lower_bounds), n_multistarts)
    found_starts = 0
    while true
        # QuasiMonteCarlo is deterministic, so for sufficiently few start-guesses we can end up in a never ending
        # loop. To sidestep this if less than 10 starts are left numbers are generated from the uniform distribution
        if n_multistarts - found_starts > 10
            _samples = QuasiMonteCarlo.sample(n_multistarts - found_starts, lower_bounds, upper_bounds, sampling_method)
        else
            _samples = Matrix{Float64}(undef, length(lower_bounds), n_multistarts - found_starts)
            for i in 1:(n_multistarts - found_starts)
                _samples[:, i] .= [rand() * (upper_bounds[j] - lower_bounds[j]) + lower_bounds[j] for j in eachindex(lower_bounds)]
            end
        end

        # Account for potential initalisation priors
        for (θ_name, _dist) in prior_info.initialisation_distribution
            _i = findfirst(x -> x == θ_name, θ_names)
            _lb, _ub = get_bounds_prior(θ_name, petab_problem)
            _prior_samples = sample_from_prior(n_multistarts - found_starts, _dist, _lb, _ub)
            transform_prior_samples!(_prior_samples, θ_name, petab_problem)
            _samples[_i, :] .= _prior_samples
        end

        for i in 1:size(_samples)[2]
            _p = _samples[:, i]
            _cost = petab_problem.compute_cost(_p)
            if allow_inf_for_startguess == true
                found_starts += 1
                startguesses[:, found_starts] .= _p
            elseif !isinf(_cost)
                found_starts += 1
                startguesses[:, found_starts] .= _p
            end
        end
        verbose == true && @printf("Found %d of %d multistarts\n", found_starts, n_multistarts)
        if found_starts == n_multistarts
            break
        end
    end

    return startguesses
end


function get_bounds_prior(θ_name::Symbol,
                          petab_problem::PEtabODEProblem)::Vector{Float64}

    @unpack prior_info, lower_bounds, upper_bounds = petab_problem
    i = findfirst(x -> x == θ_name, petab_problem.θ_names)
    if prior_info.prior_on_parameter_scale[θ_name] == true
        return [lower_bounds[i], upper_bounds[i]]
    end

    # Here the prior is on the linear scale, while the bounds are on parameter
    # scale so they must be transformed
    scale = petab_problem.compute_cost.parameter_info.parameter_scale[i]
    lower_bound = transform_θ_element(lower_bounds[i], scale, reverse_transform=false)
    upper_bound = transform_θ_element(upper_bounds[i], scale, reverse_transform=false)
    return [lower_bound, upper_bound]
end


function transform_prior_samples!(samples::Vector{Float64},
                                  θ_name::Symbol,
                                  petab_problem::PEtabODEProblem)::Nothing

    @unpack prior_info, lower_bounds, upper_bounds = petab_problem
    i = findfirst(x -> x == θ_name, petab_problem.θ_names)
    if prior_info.prior_on_parameter_scale[θ_name] == true
        return nothing
    end

    # Here the prior is on the linear scale, while the bounds are on parameter
    # so the prior samples are linear, thus they must be transformed back to 
    # parmeter scale for the parameter estimation
    scale = petab_problem.compute_cost.parameter_info.parameter_scale[i]
    for i in eachindex(samples)
        samples[i] = transform_θ_element.(samples[i], scale, reverse_transform=true)
    end

    return nothing
end


"""
    sample_from_prior(n_samples::Int64,
                      dist::Distribution{Univariate, Continuous},
                      lower_bound::Float64,
                      upper_bound::Float64)::Vector{Float64}

Draw `n_samples` from distribituion `dist` truncated at `lower_bound` and `upper_bound`.

Used for generating start-guesses for calibration when the user has provided an initialisation prior.
"""
function sample_from_prior(n_samples::Int64,
                           dist::Distribution{Univariate, Continuous},
                           lower_bound::Float64,
                           upper_bound::Float64)::Vector{Float64}

    dist_sample = truncated(dist, lower=lower_bound, upper=upper_bound)
    samples = rand(dist_sample, n_samples)
    println("samples = ", samples)
    return samples
end


function save_partial_results(path_save_res::String,
                              path_save_parameters::String,
                              path_save_trace::Union{String, Nothing},
                              res::PEtabOptimisationResult,
                              θ_names::Vector{Symbol},
                              i::Int64)::Nothing

    df_save_res = DataFrame(fmin=res.fmin,
                          alg=String(res.alg),
                          n_iterations = res.n_iterations,
                          run_time = res.runtime,
                          converged=string(res.converged),
                          Start_guess=i)
    df_save_parameters = DataFrame(Matrix(res.xmin'), θ_names)
    df_save_parameters[!, "Start_guess"] = [i]
    CSV.write(path_save_res, df_save_res, append=isfile(path_save_res))
    CSV.write(path_save_parameters, df_save_parameters, append=isfile(path_save_parameters))

    if !isnothing(path_save_trace)
        df_save_trace = DataFrame(Matrix(reduce(vcat, res.xtrace')), θ_names)
        df_save_trace[!, "f_trace"] = res.ftrace
        df_save_trace[!, "Start_guess"] = repeat([i], length(res.ftrace))
        CSV.write(path_save_trace, df_save_trace, append=isfile(path_save_trace))
    end
    return nothing
end


function _calibrate_model_multistart(petab_problem::PEtabODEProblem,
                                     alg,
                                     n_multistarts,
                                     dir_save,
                                     sampling_method,
                                     options,
                                     save_trace::Bool)::PEtabMultistartOptimisationResult

    if isnothing(dir_save)
        path_save_x0, path_save_res, path_save_trace = nothing, nothing, nothing
    else
        !isdir(dir_save) && mkpath(dir_save)
        _i = 1
        while true
            path_save_x0 = joinpath(dir_save, "Start_guesses" * string(_i) * ".csv")
            if !isfile(path_save_x0)
                break
            end
            _i += 1
        end
        path_save_x0 = joinpath(dir_save, "Start_guesses" * string(_i) * ".csv")
        path_save_res = joinpath(dir_save, "Optimisation_results" * string(_i) * ".csv")
        path_save_parameters = joinpath(dir_save, "Best_parameters" * string(_i) * ".csv")
        if save_trace == true
            path_save_trace = joinpath(dir_save, "Trace" * string(_i) * ".csv")
        else
            path_save_trace = nothing
        end
    end

    startguesses = generate_startguesses(petab_problem, n_multistarts; sampling_method=sampling_method)
    if !isnothing(path_save_x0)
        startguessesDf = DataFrame(Matrix(startguesses)', petab_problem.θ_names)
        startguessesDf[!, "Start_guess"] = 1:size(startguessesDf)[1]
        CSV.write(path_save_x0, startguessesDf)
    end

    _res = Vector{PEtabOptimisationResult}(undef, n_multistarts)
    for i in 1:n_multistarts
        _p0 = startguesses[:, i]
        _res[i] = calibrate_model(petab_problem, _p0, alg, save_trace=save_trace, options=options)
        if !isnothing(path_save_res)
            save_partial_results(path_save_res, path_save_parameters, path_save_trace, _res[i], petab_problem.θ_names, i)
        end
    end

    res_best = _res[argmin([isnan(_res[i].fmin) ? Inf : _res[i].fmin for i in eachindex(_res)])]
    fmin = res_best.fmin
    xmin = res_best.xmin
    sampling_method_str = string(sampling_method)[1:findfirst(x -> x == '(', string(sampling_method))][1:end-1]
    results = PEtabMultistartOptimisationResult(xmin,
                                                petab_problem.θ_names,
                                                fmin,
                                                n_multistarts,
                                                res_best.alg,
                                                sampling_method_str,
                                                dir_save,
                                                _res)
    return results
end