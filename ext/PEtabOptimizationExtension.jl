module PEtabOptimizationExtension

using CSV
using ModelingToolkit
using SciMLBase
using QuasiMonteCarlo
using Random
using Printf
using YAML
using PEtab
using Optimization


function PEtab.OptimizationProblem(petab_problem::PEtabODEProblem;
                                   interior_point_alg::Bool=false,
                                   box_constraints::Bool=true)::Optimization.OptimizationProblem

    optimization_problem = get_optimization_problem(petab_problem;
                                                    interior_point=interior_point_alg,
                                                    box_constraints=box_constraints)
    return optimization_problem
end


function get_optimization_problem(petab_problem::PEtabODEProblem;
                                  interior_point::Bool=false,
                                  box_constraints::Bool=true)::Optimization.OptimizationProblem

    # First build the OptimizationFunction with PEtab.jl objective, gradient and Hessian
    _f = (u, p) -> petab_problem.compute_cost(u)
    _∇f = (G, u, p) -> petab_problem.compute_gradient!(G, u)
    _Δf = (H, u, p) -> petab_problem.compute_hessian!(H, u)
    constraints = (res, x, p) -> res .= 0.0
    constraints_J = (res, x, p) -> res .= 0
    constraints_H = (res, x, p) -> begin
        for i in eachindex(res)
            res[i] .= 0.0
        end
    end
    if interior_point == true
        optimization_function = Optimization.OptimizationFunction(_f;
                                                                  grad = _∇f,
                                                                  hess = _Δf,
                                                                  cons = constraints,
                                                                  cons_j = constraints_J,
                                                                  cons_h = constraints_H,
                                                                  syms=petab_problem.θ_names)
    else
        optimization_function = Optimization.OptimizationFunction(_f;
                                                                  grad = _∇f,
                                                                  hess = _Δf,
                                                                  syms=petab_problem.θ_names)
    end

    # Build the optimisation problem
    @unpack lower_bounds, upper_bounds = petab_problem
    u0 = deepcopy(petab_problem.θ_nominalT)
    if interior_point == true
        lcons = fill(-Inf, length(petab_problem.θ_names))
        ucons = fill(Inf, length(petab_problem.θ_names))
    else
        lcons, ucons = nothing, nothing
    end

    if box_constraints == false
        lower_bounds, upper_bounds = nothing, nothing
    end
    optimization_problem = Optimization.OptimizationProblem(optimization_function, u0,
                                                            lb=lower_bounds, ub=upper_bounds,
                                                            lcons=lcons, ucons=ucons)

    return optimization_problem
end


function PEtab.calibrate_model(optimization_problem::Optimization.OptimizationProblem,
                               p0::Vector{Float64},
                               alg;
                               kwargs...)::PEtab.PEtabOptimisationResult

    optimization_problem.u0 .= deepcopy(p0)

    # save_trace not availble option
    ftrace = Vector{Float64}(undef, 0)
    xtrace = Vector{Vector{Float64}}(undef, 0)

    # Create a runnable function taking parameter as input
    n_iterations = 0
    local n_iterations, fmin, xmin, converged, runtime
    try
        runtime = @elapsed sol = solve(optimization_problem, alg)
        fmin = sol.objective
        xmin = sol.u
        converged = sol.retcode
    catch
        n_iterations = 0
        fmin = NaN
        xmin = similar(p0) .* NaN
        converged = :Code_crashed
        runtime = NaN
    end
    alg_used = :Optimization_package

    return PEtabOptimisationResult(alg_used,
                                   xtrace,
                                   ftrace,
                                   n_iterations,
                                   fmin,
                                   deepcopy(p0),
                                   xmin,
                                   optimization_problem.f.syms,
                                   converged,
                                   runtime)
end


function PEtab.calibrate_model_multistart(optimization_problem::Optimization.OptimizationProblem,
                                          petab_problem::PEtabODEProblem,
                                          alg,
                                          n_multistarts::Signed,
                                          dir_save::Union{Nothing, String};
                                          sampling_method::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                          sample_from_prior::Bool=true,
                                          seed::Union{Nothing, Integer}=nothing,
                                          kwargs...)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end

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

    startguesses = generate_startguesses(petab_problem, n_multistarts; sampling_method=sampling_method, sample_from_prior=sample_from_prior)
    if !isnothing(path_save_x0)
        startguessesDf = DataFrame(Matrix(startguesses)', petab_problem.θ_names)
        startguessesDf[!, "Start_guess"] = 1:size(startguessesDf)[1]
        CSV.write(path_save_x0, startguessesDf)
    end

    _res = Vector{PEtabOptimisationResult}(undef, n_multistarts)
    for i in 1:n_multistarts
        _p0 = startguesses[:, i]
        _res[i] = calibrate_model(optimization_problem, _p0, alg; kwargs...)
        if !isnothing(path_save_res)
            save_partial_results(path_save_res, path_save_parameters, path_save_trace, _res[i], petab_problem.θ_names, i)
        end
    end

    res_best = _res[argmin([isnan(_res[i].fmin) ? Inf : _res[i].fmin for i in eachindex(_res)])]
    fmin = res_best.fmin
    xmin = res_best.xmin
    sampling_method_str = string(sampling_method)[1:findfirst(x -> x == '(', string(sampling_method))][1:end-1]
    results = PEtabMultistartOptimisationResult(xmin,
                                                optimization_problem.f.syms,
                                                fmin,
                                                n_multistarts,
                                                res_best.alg,
                                                sampling_method_str,
                                                dir_save,
                                                _res)
    return results
end

end