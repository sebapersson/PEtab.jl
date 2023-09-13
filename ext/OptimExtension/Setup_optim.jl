#=
    Optim wrapper 
=#


function PEtab.callibrateModelMultistart(petabProblem::PEtabODEProblem, 
                                         alg::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton}, 
                                         nMultiStarts::Signed, 
                                         dirSave::Union{Nothing, String};
                                         samplingMethod::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                         options::Optim.Options=Optim.Options(iterations = 1000,
                                                                              show_trace = false,
                                                                              allow_f_increases=true,
                                                                              successive_f_tol = 3,
                                                                              f_tol=1e-8,
                                                                              g_tol=1e-6,
                                                                              x_tol=0.0), 
                                         seed::Union{Nothing, Integer}=nothing, 
                                         saveTrace::Bool=false)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end
    res = PEtab._multistartModelCallibration(petabProblem, alg, nMultiStarts, dirSave, samplingMethod, options, saveTrace)
    return res
end


function PEtab.callibrateModel(petabProblem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton}; 
                               saveTrace::Bool=false, 
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
                            store_trace=saveTrace, 
                            trace_simplex=options.trace_simplex, 
                            show_trace=options.show_trace, 
                            extended_trace=saveTrace, 
                            show_every=options.show_every, 
                            callback=options.callback, 
                            time_limit=options.time_limit)

    # Create a runnable function taking parameter as input                            
    optimProblem =  createOptimProblem(petabProblem, alg, options=_options)
    local nIterations, fMin, xMin, converged, runTime, fTrace, xTrace
    try
        res = optimProblem(p0)
        nIterations = Optim.iterations(res)
        fMin = Optim.minimum(res)
        xMin = Optim.minimizer(res)
        converged = Optim.converged(res)
        runTime = res.time_run # In seconds 
        if saveTrace == true
            fTrace = Optim.f_trace(res)
            xTrace = Optim.x_trace(res)
        else
            fTrace = Vector{Float64}(undef, 0)
            xTrace = Vector{Vector{Float64}}(undef, 0)
        end
    catch
        nIterations = 0
        fMin = NaN
        xMin = similar(p0) .* NaN
        fTrace = Vector{Float64}(undef, 0)
        xTrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runTime = NaN
    end

    if typeof(alg) <: Optim.IPNewton
        algUsed = :Optim_IPNewton
    elseif typeof(alg) <: Optim.BFGS
        algUsed = :Optim_BFGS
    elseif typeof(alg) <: Optim.LBFGS
        algUsed = :Optim_LBFGS
    end

    return PEtabOptimisationResult(algUsed,
                                   xTrace, 
                                   fTrace, 
                                   nIterations, 
                                   fMin, 
                                   _p0,
                                   xMin, 
                                   converged, 
                                   runTime)
end


"""
    createOptimProb(petabProblem::PEtabODEProblem,
                    optimAlg;
                    hessianUse::Symbol=:blockAutoDiff)

For a PeTab model optimization struct (petabProblem) create an Optim optmization (evalOptim)
function using as optimAlg IPNewton (interior point Newton) or LBFGS, BFGS, ConjugateGradient.
For IPNewton the hessian is computed via eiter autoDiff (:autoDiff), or approximated
with blockAutoDiff (:blockAutoDiff). All optimizer struct can take their default
arguments, for example, LBFGS(linesearch = LineSearches.HagerZhang()) is a valid
argument for LBFGS.
"""
function createOptimProblem(petabProblem::PEtabODEProblem,
                            optimAlg;
                            options=Optim.Options(iterations = 1000,
                                                  show_trace = false,
                                                  allow_f_increases=true,
                                                  successive_f_tol = 3,
                                                  f_tol=1e-8,
                                                  g_tol=1e-6,
                                                  x_tol=0.0))

    if typeof(optimAlg) <: IPNewton
        return createOptimInteriorNewton(petabProblem, optimAlg, options)
    elseif typeof(optimAlg) <: LBFGS || typeof(optimAlg) <: BFGS || typeof(optimAlg) <: ConjugateGradient
        return createOptimFminbox(petabProblem, optimAlg, options)
    else
        @error "optimAlg $optimAlg is not supported - supported methods are IPNewton, ConjugateGradient, LBFGS and BFGS"
    end
end


"""
    createOptimInteriorNewton(petabProblem::PEtabODEProblem;
                              hessianUse::Symbol=:blockAutoDiff)

For a PeTab model optimization struct (petabProblem) create an Optim interior point Newton
function struct where the hessian is computed via eiter autoDiff (:autoDiff), or approximated
with blockAutoDiff (:blockAutoDiff).
"""
function createOptimInteriorNewton(petabProblem::PEtabODEProblem,
                                   optimAlg,
                                   options)

    lowerBounds = petabProblem.lowerBounds
    upperBounds = petabProblem.upperBounds

    nParam = length(lowerBounds)
    x0 = zeros(Float64, nParam)
    df = TwiceDifferentiable(petabProblem.computeCost, petabProblem.computeGradient!, petabProblem.computeHessian!, x0)
    dfc = TwiceDifferentiableConstraints(lowerBounds, upperBounds)

    evalOptim = (p0) -> begin
                            # Move points within bounds
                            iBelow = p0 .<= petabProblem.lowerBounds
                            iAbove = p0 .>= petabProblem.upperBounds
                            p0[iBelow] .= petabProblem.lowerBounds[iBelow] .+ 0.001
                            p0[iAbove] .= petabProblem.upperBounds[iAbove] .- 0.001
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
    createOptimFminbox(petabProblem::PEtabODEProblem;
                       lineSearch=LineSearches.HagerZhang())

For a PeTab model optimization struct (petabProblem) create an Optim Fminbox optimizer where the
inner optimizer is either LBFGS or BFGS using lineSearch.
"""
function createOptimFminbox(petabProblem::PEtabODEProblem,
                            optimAlg,
                            options)

    lowerBounds = petabProblem.lowerBounds
    upperBounds = petabProblem.upperBounds

    evalOptim = (p0) -> Optim.optimize(petabProblem.computeCost,
                                       petabProblem.computeGradient!,
                                       lowerBounds,
                                       upperBounds,
                                       p0,
                                       Fminbox(optimAlg),
                                       options)

    return evalOptim
end