#=
    Optim wrapper 
=#

calibrate_model_multistart
function PEtab.calibrate_model_multistart(petab_problem::PEtabODEProblem, 
                                        alg::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton}, 
                                        n_multistarts::Signed, 
                                        dir_save::Union{Nothing, String};
                                        sampling_method::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                        options::Optim.Options=Optim.Options(iterations = 1000,
                                                                             show_trace = false,
                                                                             allow_f_increases=true,
                                                                             successive_f_tol = 3,
                                                                             f_tol=1e-8,
                                                                             g_tol=1e-6,
                                                                             x_tol=0.0), 
                                        seed::Union{Nothing, Integer}=nothing, 
                                        save_trace::Bool=false)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end
    res = PEtab._multistartModelCallibration(petab_problem, alg, n_multistarts, dir_save, sampling_method, options, save_trace)
    return res
end


function PEtab.calibrate_model(petab_problem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton}; 
                               save_trace::Bool=false, 
                               options=Optim.Options(iterations = 1000,
                                                     show_trace = false,
                                                     allow_f_increases=true,
                                                     successive_f_tol = 3,
                                                     f_tol=1e-8,
                                                     g_tol=1e-6,
                                                     x_tol=0.0))::PEtab.PEtabOptimisationResult

    _p0 = deepcopy(p0)                                                     

    # Whether or not trace should be stored is an input argument to callibrate model, 
    # but as Optim.Options is inmutable this is how I have to set the value :(
    _options = Optim.Options(x_abstol=options.x_abstol, 
                            x_reltol=options.x_reltol, 
                            f_abstol=options.f_abstol, 
                            f_reltol=options.f_reltol, 
                            g_abstol=options.g_abstol, 
                            g_reltol=options.g_reltol,
                            outer_x_abstol=options.outer_x_abstol, 
                            outer_x_reltol=options.outer_x_reltol, 
                            outer_f_abstol=options.outer_f_abstol, 
                            outer_f_reltol=options.outer_f_reltol, 
                            outer_g_abstol=options.outer_g_abstol, 
                            outer_g_reltol=options.outer_g_reltol, 
                            f_calls_limit=options.f_calls_limit, 
                            g_calls_limit=options.g_calls_limit, 
                            h_calls_limit=options.h_calls_limit, 
                            allow_f_increases=options.allow_f_increases, 
                            allow_outer_f_increases=options.allow_outer_f_increases, 
                            successive_f_tol=options.successive_f_tol, 
                            iterations=options.iterations, 
                            outer_iterations=options.outer_iterations, 
                            store_trace=save_trace, 
                            trace_simplex=options.trace_simplex, 
                            show_trace=options.show_trace, 
                            extended_trace=save_trace, 
                            show_every=options.show_every, 
                            callback=options.callback, 
                            time_limit=options.time_limit)

    # Create a runnable function taking parameter as input                            
    optimProblem =  createOptimProblem(petab_problem, alg, options=_options)
    local n_iterations, fmin, xmin, converged, runtime, ftrace, xtrace
    try
        res = optimProblem(p0)
        n_iterations = Optim.iterations(res)
        fmin = Optim.minimum(res)
        xmin = Optim.minimizer(res)
        converged = Optim.converged(res)
        runtime = res.time_run # In seconds 
        if save_trace == true
            ftrace = Optim.f_trace(res)
            xtrace = Optim.x_trace(res)
        else
            ftrace = Vector{Float64}(undef, 0)
            xtrace = Vector{Vector{Float64}}(undef, 0)
        end
    catch
        n_iterations = 0
        fmin = NaN
        xmin = similar(p0) .* NaN
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runtime = NaN
    end

    if typeof(alg) <: Optim.IPNewton
        algUsed = :Optim_IPNewton
    elseif typeof(alg) <: Optim.BFGS
        algUsed = :Optim_BFGS
    elseif typeof(alg) <: Optim.LBFGS
        algUsed = :Optim_LBFGS
    end

    return PEtabOptimisationResult(algUsed,
                                   xtrace, 
                                   ftrace, 
                                   n_iterations, 
                                   fmin, 
                                   _p0,
                                   xmin, 
                                   converged, 
                                   runtime)
end


"""
    createOptimProb(petab_problem::PEtabODEProblem,
                    optimAlg;
                    hessianUse::Symbol=:blockAutoDiff)

For a PeTab model optimization struct (petab_problem) create an Optim optmization (evalOptim)
function using as optimAlg IPNewton (interior point Newton) or LBFGS, BFGS, ConjugateGradient.
For IPNewton the hessian is computed via eiter autoDiff (:autoDiff), or approximated
with blockAutoDiff (:blockAutoDiff). All optimizer struct can take their default
arguments, for example, LBFGS(linesearch = LineSearches.HagerZhang()) is a valid
argument for LBFGS.
"""
function createOptimProblem(petab_problem::PEtabODEProblem,
                            optimAlg;
                            options=Optim.Options(iterations = 1000,
                                                  show_trace = false,
                                                  allow_f_increases=true,
                                                  successive_f_tol = 3,
                                                  f_tol=1e-8,
                                                  g_tol=1e-6,
                                                  x_tol=0.0))

    if typeof(optimAlg) <: IPNewton
        return createOptimInteriorNewton(petab_problem, optimAlg, options)
    elseif typeof(optimAlg) <: LBFGS || typeof(optimAlg) <: BFGS || typeof(optimAlg) <: ConjugateGradient
        return createOptimFminbox(petab_problem, optimAlg, options)
    else
        @error "optimAlg $optimAlg is not supported - supported methods are IPNewton, ConjugateGradient, LBFGS and BFGS"
    end
end


"""
    createOptimInteriorNewton(petab_problem::PEtabODEProblem;
                              hessianUse::Symbol=:blockAutoDiff)

For a PeTab model optimization struct (petab_problem) create an Optim interior point Newton
function struct where the hessian is computed via eiter autoDiff (:autoDiff), or approximated
with blockAutoDiff (:blockAutoDiff).
"""
function createOptimInteriorNewton(petab_problem::PEtabODEProblem,
                                   optimAlg,
                                   options)

    lower_bounds = petab_problem.lower_bounds
    upper_bounds = petab_problem.upper_bounds

    nParam = length(lower_bounds)
    x0 = zeros(Float64, nParam)
    df = TwiceDifferentiable(petab_problem.compute_cost, petab_problem.compute_gradient!, petab_problem.compute_hessian!, x0)
    dfc = TwiceDifferentiableConstraints(lower_bounds, upper_bounds)

    evalOptim = (p0) -> begin
                            # Move points within bounds
                            iBelow = p0 .<= petab_problem.lower_bounds
                            iAbove = p0 .>= petab_problem.upper_bounds
                            p0[iBelow] .= petab_problem.lower_bounds[iBelow] .+ 0.001
                            p0[iAbove] .= petab_problem.upper_bounds[iAbove] .- 0.001
                            df.f(p0)
                            return Optim.optimize(df,
                                                  dfc,
                                                  p0,
                                                  optimAlg,
                                                  options)
                        end

    return evalOptim
end


"""
    createOptimFminbox(petab_problem::PEtabODEProblem;
                       lineSearch=LineSearches.HagerZhang())

For a PeTab model optimization struct (petab_problem) create an Optim Fminbox optimizer where the
inner optimizer is either LBFGS or BFGS using lineSearch.
"""
function createOptimFminbox(petab_problem::PEtabODEProblem,
                            optimAlg,
                            options)

    lower_bounds = petab_problem.lower_bounds
    upper_bounds = petab_problem.upper_bounds

    evalOptim = (p0) -> Optim.optimize(petab_problem.compute_cost,
                                       petab_problem.compute_gradient!,
                                       lower_bounds,
                                       upper_bounds,
                                       p0,
                                       Fminbox(optimAlg),
                                       options)

    return evalOptim
end