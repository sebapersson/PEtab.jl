#=
    Ipopt wrapper 
=#


function PEtab.calibrate_model_multistart(petab_problem::PEtabODEProblem, 
                                          alg::IpoptOptimiser, 
                                          n_multistarts::Signed, 
                                          dir_save::Union{Nothing, String};
                                          sampling_method::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                          options::IpoptOptions=IpoptOptions(),
                                          seed::Union{Nothing, Integer}=nothing, 
                                          save_trace::Bool=false)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end
    res = PEtab._calibrate_model_multistart(petab_problem, alg, n_multistarts, dir_save, sampling_method, options, save_trace)
    return res
end


function PEtab.calibrate_model(petab_problem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::IpoptOptimiser; 
                               save_trace::Bool=false, 
                               options::IpoptOptions=IpoptOptions())::PEtab.PEtabOptimisationResult

    _p0 = deepcopy(p0)                               

    ipopt_problem, iter_arr, ftrace, xtrace = create_ipopt_problem(petab_problem, alg.LBFGS, save_trace, options)
    ipopt_problem.x = deepcopy(p0)
    
    # Create a runnable function taking parameter as input                            
    local n_iterations, fmin, xmin, converged, runtime
    try
        runtime = @elapsed sol_opt = Ipopt.IpoptSolve(ipopt_problem)
        fmin = ipopt_problem.obj_val
        xmin = ipopt_problem.x
        n_iterations = iter_arr[1]
        converged = ipopt_problem.status
    catch
        n_iterations = 0
        fmin = NaN
        xmin = similar(p0) .* NaN
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runtime = NaN
    end
    if alg.LBFGS == true
        alg_used = :Ipopt_LBFGS
    else
        alg_used = :Ipopt_user_Hessian
    end

    return PEtabOptimisationResult(alg_used,
                                   xtrace, 
                                   ftrace, 
                                   n_iterations, 
                                   fmin, 
                                   _p0,
                                   xmin, 
                                   petab_problem.θ_names,
                                   converged, 
                                   runtime)
end


function create_ipopt_problem(petab_problem::PEtabODEProblem,
                              LBFGS::Bool, 
                              save_trace::Bool,
                              options::PEtab.IpoptOptions)

    lower_bounds = petab_problem.lower_bounds
    upper_bounds = petab_problem.upper_bounds

    if LBFGS == true
        eval_hessian! = eval_h_empty
    else
        eval_hessian! = (x_arg, rows, cols, obj_factor, lambda, values) -> eval_h(x_arg, rows, cols, obj_factor, lambda, values, n_parameters, petab_problem.compute_hessian!)
    end

    n_parameters = length(lower_bounds)
    eval_gradient! = (xArg, grad) -> petab_problem.compute_gradient!(grad, xArg)

    m = 0
    n_parameters_hessian = Int(n_parameters*(n_parameters + 1) / 2)
    # No inequality constraints assumed (to be added in the future)
    g_L = Float64[]
    g_U = Float64[] 
    prob = Ipopt.CreateIpoptProblem(n_parameters, 
                                    lower_bounds, 
                                    upper_bounds, 
                                    m, # No constraints
                                    g_L, # No constraints
                                    g_U, # No constraints
                                    0, # No constraints
                                    n_parameters_hessian, 
                                    petab_problem.compute_cost, 
                                    eval_g, # No constraints 
                                    eval_gradient!,
                                    eval_jac_g, 
                                    eval_hessian!)
    # Ipopt does not allow the iteration count to be stored directly. Thus the iteration is stored in an arrary which 
    # is sent into the Ipopt callback function. 
    iter_arr = ones(Int64, 1) .* 20
    ftrace = Vector{Float64}(undef, 0)
    xtrace = Vector{Vector{Float64}}(undef, 0)
    _intermediate = (alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials) -> intermediate_ipopt(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, iter_arr, prob, save_trace, ftrace, xtrace)
    Ipopt.SetIntermediateCallback(prob, _intermediate) # Allow iterations to be retrevied (see above) 

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

    return prob, iter_arr, ftrace, xtrace
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
                n_parameters::Integer, 
                eval_hessian!::Function)

    idx::Int32 = 0
    if values === nothing
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        @inbounds for row in 1:n_parameters
            for col in 1:row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        # Symmetric, fill only lower left matrix (values)
        hessian = zeros(Float64, n_parameters, n_parameters)
        eval_hessian!(hessian, xArg)
        idx = 1
        @inbounds for row in 1:n_parameters
            for col in 1:row
                values[idx] = hessian[row, col] * obj_factor
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
                            iter_arr, 
                            ipopt_prob, 
                            save_trace::Bool, 
                            ftrace::Vector{Float64}, 
                            xtrace::Vector{Vector{Float64}})
    iter_arr[1] = Int(iter_count)
    if save_trace == true
        push!(ftrace, obj_value)
        push!(xtrace, deepcopy(ipopt_prob.x))
    end

    return true 
end