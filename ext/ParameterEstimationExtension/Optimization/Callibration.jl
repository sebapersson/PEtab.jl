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
                                                     x_tol=0.0))::PEtabOptimisationResult

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
                                   xMin, 
                                   converged, 
                                   runTime)
end
function PEtab.callibrateModel(petabProblem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::Fides; 
                               saveTrace::Bool=false, 
                               options=py"{'maxiter' : 1000}"o)::PEtabOptimisationResult

    if saveTrace == true                         
        @warn "For Fides the x and f trace cannot currently be saved (we are working on it)" maxlog=10     
    end          

    runFides = createFidesProblem(petabProblem, alg, options=options)

    # Create a runnable function taking parameter as input                            
    local nIterations, fMin, xMin, converged, runTime, fTrace, xTrace
    try
        runTime = @elapsed res, nIterations, converged = runFides(p0)
        fMin = res[1]
        xMin = res[2]
        fTrace = Vector{Float64}(undef, 0)
        xTrace = Vector{Vector{Float64}}(undef, 0)
    catch
        nIterations = 0
        fMin = NaN
        xMin = similar(p0) .* NaN
        fTrace = Vector{Float64}(undef, 0)
        xTrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runTime = NaN
    end
    algUsed = :Fides

    return PEtabOptimisationResult(algUsed,
                                   xTrace, 
                                   fTrace, 
                                   nIterations, 
                                   fMin, 
                                   xMin, 
                                   converged, 
                                   runTime)
end
function PEtab.callibrateModel(petabProblem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::IpoptOptimiser; 
                               saveTrace::Bool=false, 
                               options::IpoptOptions=IpoptOptions())::PEtabOptimisationResult


    ipoptProblem, iterArr, fTrace, xTrace = createIpoptProblem(petabProblem, alg.approximateHessian, saveTrace, options)
    ipoptProblem.x = deepcopy(p0)
    
    # Create a runnable function taking parameter as input                            
    local nIterations, fMin, xMin, converged, runTime
    try
        runTime = @elapsed sol_opt = Ipopt.IpoptSolve(ipoptProblem)
        fMin = ipoptProblem.obj_val
        xMin = ipoptProblem.x
        nIterations = iterArr[1]
        converged = ipoptProblem.status
    catch
        nIterations = 0
        fMin = NaN
        xMin = similar(p0) .* NaN
        fTrace = Vector{Float64}(undef, 0)
        xTrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runTime = NaN
    end
    if alg.approximateHessian == true
        algUsed = :Ipopt_LBFGS
    else
        algUsed = :Ipopt_user_Hessian
    end

    return PEtabOptimisationResult(algUsed,
                                   xTrace, 
                                   fTrace, 
                                   nIterations, 
                                   fMin, 
                                   xMin, 
                                   converged, 
                                   runTime)
end


function generateStartGuesses(petabProblem::PEtabODEProblem,
                              samplingMethod::T,
                              nOptimisationStarts::Int) where T <: QuasiMonteCarlo.SamplingAlgorithm

    # Nothing prevents the user from sending in a parameter vector with zero parameters
    if length(petabProblem.lowerBounds) == 0
        return nothing
    end

    # Return a random number sampled from uniform distribution
    if nOptimisationStarts == 1
        return [rand() * (petabProblem.upperBounds[i] - petabProblem.lowerBounds[i]) + petabProblem.lowerBounds[i] for i in eachindex(petabProblem.lowerBounds)]
    end

    return QuasiMonteCarlo.sample(nOptimisationStarts, petabProblem.lowerBounds, petabProblem.upperBounds, samplingMethod)
end