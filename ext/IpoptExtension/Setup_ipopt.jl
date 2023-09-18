#=
    Ipopt wrapper 
=#


function PEtab.calibrateModelMultistart(petabProblem::PEtabODEProblem, 
                                        alg::IpoptOptimiser, 
                                        nMultiStarts::Signed, 
                                        dirSave::Union{Nothing, String};
                                        samplingMethod::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                        options::IpoptOptions=IpoptOptions(),
                                        seed::Union{Nothing, Integer}=nothing, 
                                        saveTrace::Bool=false)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end
    res = PEtab._multistartModelCallibration(petabProblem, alg, nMultiStarts, dirSave, samplingMethod, options, saveTrace)
    return res
end


function PEtab.calibrateModel(petabProblem::PEtabODEProblem, 
                              p0::Vector{Float64},
                              alg::IpoptOptimiser; 
                              saveTrace::Bool=false, 
                              options::IpoptOptions=IpoptOptions())::PEtab.PEtabOptimisationResult

    _p0 = deepcopy(p0)                               

    ipoptProblem, iterArr, fTrace, xTrace = createIpoptProblem(petabProblem, alg.LBFGS, saveTrace, options)
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
    if alg.LBFGS == true
        algUsed = :Ipopt_LBFGS
    else
        algUsed = :Ipopt_user_Hessian
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


function createIpoptProblem(petabProblem::PEtabODEProblem,
                            LBFGS::Bool, 
                            saveTrace::Bool,
                            options::PEtab.IpoptOptions)

    lowerBounds = petabProblem.lowerBounds
    upperBounds = petabProblem.upperBounds

    if LBFGS == true
        evalHessian = eval_h_empty
    else
        evalHessian = (x_arg, rows, cols, obj_factor, lambda, values) -> eval_h(x_arg, rows, cols, obj_factor, lambda, values, nParam, petabProblem.computeHessian!)
    end

    nParam = length(lowerBounds)
    evalGradFUse = (xArg, grad) -> petabProblem.computeGradient!(grad, xArg)


    m = 0
    nParamHess = Int(nParam*(nParam + 1) / 2)
    # No inequality constraints assumed (to be added in the future)
    g_L = Float64[]
    g_U = Float64[] 
    prob = Ipopt.CreateIpoptProblem(nParam, 
                                    lowerBounds, 
                                    upperBounds, 
                                    m, # No constraints
                                    g_L, # No constraints
                                    g_U, # No constraints
                                    0, # No constraints
                                    nParamHess, 
                                    petabProblem.computeCost, 
                                    eval_g, # No constraints 
                                    evalGradFUse,
                                    eval_jac_g, 
                                    evalHessian)
    # Ipopt does not allow the iteration count to be stored directly. Thus the iteration is stored in an arrary which 
    # is sent into the Ipopt callback function. 
    iterArr = ones(Int64, 1) .* 20
    fTrace = Vector{Float64}(undef, 0)
    xTrace = Vector{Vector{Float64}}(undef, 0)
    intermediateUse = (alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials) -> intermediate_ipopt(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, iterArr, prob, saveTrace, fTrace, xTrace)
    Ipopt.SetIntermediateCallback(prob, intermediateUse) # Allow iterations to be retrevied (see above) 

    if LBFGS == true
        Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    end 

    # Set options 
    Ipopt.AddIpoptIntOption(prob, "print_level", options.print_level)
    Ipopt.AddIpoptIntOption(prob, "max_iter", options.max_iter)
    Ipopt.AddIpoptNumOption(prob, "tol", options.tol)
    Ipopt.AddIpoptNumOption(prob, "acceptable_tol", options.acceptable_tol)
    Ipopt.AddIpoptNumOption(prob, "max_wall_time", options.max_wall_time)
    Ipopt.AddIpoptNumOption(prob, "acceptable_obj_change_tol", options.acceptable_obj_change_tol)

    return prob, iterArr, fTrace, xTrace
end


"""
    eval_h(x_arg::Vector{Float64}, 
           rows::Vector{Int32}, 
           cols::Vector{Int32}, 
           obj_factor::Float64, 
           lambda::Vector{Float64}, 
           values::Union{Nothing,Vector{Float64}}, 
           n_param, 
           evalH::Function)

    Helper function computing the hessian for Ipopt (all correct input arguments). 
    The actual hessian is computed via the evalH function, and this function must 
    be on the format evalH(hessian, paramVec).

    Ipopt supports sparse hessians, but at the moment a dense hessian is assumed.
"""
function eval_h(xArg::Vector{Float64}, 
                rows::Vector{Int32}, 
                cols::Vector{Int32}, 
                obj_factor::Float64, 
                lambda::Vector{Float64}, 
                values::Union{Nothing,Vector{Float64}}, 
                nParam::Integer, 
                evalH::Function)

    idx::Int32 = 0
    if values === nothing
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        @inbounds for row in 1:nParam
            for col in 1:row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        # Symmetric, fill only lower left matrix (values)
        hessianMat = zeros(nParam, nParam)
        evalH(hessianMat, xArg)
        idx = 1
        @inbounds for row in 1:nParam
            for col in 1:row
                values[idx] = hessianMat[row, col] * obj_factor
                idx += 1
            end
        end
    end
    return
end


# In case of of BFGS gradient provide Ipopt with empty hessian struct.
function eval_h_empty(x_arg::Vector{Float64}, 
                      rows::Vector{Int32}, 
                      cols::Vector{Int32}, 
                      obj_factor::Float64, 
                      lambda::Vector{Float64}, 
                      values::Union{Nothing,Vector{Float64}})    
    return nothing
end


# These function wraps and handles constraints. TODO: Allow optimization under inequality constraints 
function eval_jac_g(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, values::Union{Nothing,Vector{Float64}})
    return 
end
function eval_g(x::Vector{Float64}, g::Vector{Float64})
    return 
end


# Callback for Ipopt allowing number of iterations to be stored.
function intermediate_ipopt(alg_mod::Cint,
                            iter_count::Cint,
                            obj_value::Float64,
                            inf_pr::Float64,
                            inf_du::Float64,
                            mu::Float64,
                            d_norm::Float64,
                            regularization_size::Float64,
                            alpha_du::Float64,
                            alpha_pr::Float64,
                            ls_trials::Cint, 
                            iterArr, 
                            ipopt_prob, 
                            saveTrace::Bool, 
                            fTrace::Vector{Float64}, 
                            xTrace::Vector{Vector{Float64}})
    iterArr[1] = Int(iter_count)
    if saveTrace == true
        push!(fTrace, obj_value)
        push!(xTrace, deepcopy(ipopt_prob.x))
    end

    return true 
end