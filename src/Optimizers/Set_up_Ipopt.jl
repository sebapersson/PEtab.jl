"""
    createIpoptProb(petabProblem::PEtabODEProblem,
                    hessianUse::Symbol)
    
    For a PeTab model optimization struct (petabProblem) create an Ipopt optimization
    struct where the hessian is computed via eiter autoDiff (:autoDiff), approximated 
    with blockAutoDiff (:blockAutoDiff) or a LBFGS approximation (:LBFGS). 
"""
function createIpoptProb(petabProblem::PEtabODEProblem;
                         hessianUse::Symbol=:LBFGS)

    lowerBounds = petabProblem.lowerBounds
    upperBounds = petabProblem.upperBounds

    nParam = length(lowerBounds)
    if hessianUse == :autoDiff
        evalHessian = (x_arg, rows, cols, obj_factor, lambda, values) -> eval_h(x_arg, rows, cols, obj_factor, lambda, values, nParam, petabProblem.computeHessian)
    elseif hessianUse == :blockAutoDiff
        evalHessian = (x_arg, rows, cols, obj_factor, lambda, values) -> eval_h(x_arg, rows, cols, obj_factor, lambda, values, nParam, petabProblem.computeHessianBlock)
    elseif hessianUse == :GaussNewton
        evalHessian = (x_arg, rows, cols, obj_factor, lambda, values) -> eval_h(x_arg, rows, cols, obj_factor, lambda, values, nParam, petabProblem.computeHessianGN)
    elseif hessianUse == :LBFGS
        evalHessian = eval_h_empty
    else
        println("Error : For Ipopt hessianUse options are :autoDiff, :blockAutoDiff, :GaussNewton, :LBFGS, not $hessianUse")
    end

    # Of course Ipopt and Optim accept the gradient in different order 
    if hessianUse == :GaussNewton
        evalGradFUse = (xArg, grad) -> petabProblem.computeGradientForwardEquations(grad, xArg)
    else
        evalGradFUse = (xArg, grad) -> petabProblem.computeGradientAutoDiff(grad, xArg)
    end

    # Ipopt does not allow the iteration count to be stored directly. Thus the iteration is stored in an arrary which 
    # is sent into the Ipopt callback function. 
    iterArr = ones(Int64, 1) .* 20
    intermediateUse = (alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials) -> intermediate(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, iterArr)

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
    Ipopt.SetIntermediateCallback(prob, intermediateUse) # Allow iterations to be retrevied (see above) 

    if hessianUse == :LBFGS
        Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    end 
    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    Ipopt.AddIpoptIntOption(prob, "max_iter", 1000)

    return prob, iterArr
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
function intermediate(alg_mod::Cint,
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
                      iterArr)
    iterArr[1] = Int(iter_count)

    return true 
end



