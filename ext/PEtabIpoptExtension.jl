module PEtabIpoptExtension

using Ipopt
import QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm
import Random
using Catalyst: @unpack
using ComponentArrays
using PEtab

function PEtab.calibrate_multistart(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, alg::IpoptOptimizer, nmultistarts;
        nprocs = 1, save_trace = false, dirsave = nothing, sample_prior = true,
        sampling_method = LatinHypercubeSample(), init_weight = nothing, init_bias = nothing,
        options::Union{Nothing, IpoptOptions} = nothing,
    )::PEtab.PEtabMultistartResult
    options = isnothing(options) ? IpoptOptions() : options

    return PEtab._calibrate_multistart(
        rng, prob, alg, nmultistarts, dirsave, sampling_method, options, sample_prior,
        save_trace, nprocs, init_weight, init_bias
    )
end

function PEtab.calibrate(prob::PEtabODEProblem,
                         x::Union{Vector{<:AbstractFloat}, ComponentArray},
                         alg::IpoptOptimizer; save_trace::Bool = false,
                         options::IpoptOptions = IpoptOptions())::PEtab.PEtabOptimisationResult
    xstart = x |> collect
    ipopt_prob, iters, ftrace, xtrace = _get_ipopt_prob(prob, alg.LBFGS, save_trace,
                                                        options)
    # Ipopt mutates input vector
    ipopt_prob.x = xstart |> deepcopy

    # Create a runnable function taking parameter as input
    local niterations, fmin, _xmin, converged, runtime, sol_ipopt
    try
        runtime = @elapsed sol_ipopt = Ipopt.IpoptSolve(ipopt_prob)
        fmin = ipopt_prob.obj_val
        _xmin = ipopt_prob.x
        niterations = iters[1]
        converged = ipopt_prob.status
    catch
        niterations = 0
        fmin = NaN
        _xmin = similar(xstart) .* NaN
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Optmisation_failed
        runtime = NaN
        sol_ipopt = nothing
    end
    xnames_ps = propertynames(prob.xnominal_transformed)
    xstart = ComponentArray(; (xnames_ps .=> xstart)...)
    xmin = ComponentArray(; (xnames_ps .=> _xmin)...)

    if alg.LBFGS == true
        alg_used = :Ipopt_LBFGS
    else
        alg_used = :Ipopt_user_Hessian
    end
    return PEtabOptimisationResult(xmin, fmin, xstart, alg_used, niterations, runtime,
                                   xtrace, ftrace, converged, sol_ipopt)
end

function _get_ipopt_prob(prob::PEtabODEProblem, LBFGS::Bool, save_trace::Bool,
                         options::PEtab.IpoptOptions)
    @unpack lower_bounds, upper_bounds = prob
    lb = lower_bounds |> collect
    ub = upper_bounds |> collect
    nparameters = length(lb)

    # Need to wrap Hessian to Ipopt format (quite a bit of work)
    if LBFGS == true
        eval_hess! = _hess_empty!
    else
        eval_hess! = (x, rows, cols, obj_factor, λ, values) -> _hess!(x, rows, cols,
                                                                      obj_factor, λ,
                                                                      values, nparameters,
                                                                      prob.hess!)
    end
    eval_grad! = (x, grad) -> prob.grad!(grad, x)

    m = 0
    nparameters_hessian = Int(nparameters * (nparameters + 1) / 2)
    # No inequality constraints assumed (can be added in future)
    g_l = Float64[]
    g_u = Float64[]
    ipopt_prob = Ipopt.CreateIpoptProblem(nparameters, lb, ub, m, g_l, g_u, 0,
                                          nparameters_hessian, prob.nllh, eval_g!,
                                          eval_grad!, eval_jac_g!, eval_hess!)
    # Ipopt does not allow the iteration count to be stored directly. Thus the iteration
    # is stored in an array which is updated in the Ipopt callback
    # is sent into the Ipopt callback function.
    iters = ones(Int64, 1)
    ftrace = Vector{Float64}(undef, 0)
    xtrace = Vector{Vector{Float64}}(undef, 0)
    _intermediate = (alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
    regularization_size, alpha_du, alpha_pr, ls_trials) -> begin
        intermediate_ipopt(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
                           regularization_size, alpha_du, alpha_pr, ls_trials, iters,
                           ipopt_prob, save_trace, ftrace, xtrace)
    end
    Ipopt.SetIntermediateCallback(ipopt_prob, _intermediate)

    # Ipopt problem options
    if LBFGS == true
        Ipopt.AddIpoptStrOption(ipopt_prob, "hessian_approximation", "limited-memory")
    end
    Ipopt.AddIpoptIntOption(ipopt_prob, "print_level", options.print_level)
    Ipopt.AddIpoptIntOption(ipopt_prob, "max_iter", options.max_iter)
    Ipopt.AddIpoptNumOption(ipopt_prob, "tol", options.tol)
    Ipopt.AddIpoptNumOption(ipopt_prob, "acceptable_tol", options.acceptable_tol)
    Ipopt.AddIpoptNumOption(ipopt_prob, "max_wall_time", options.max_wall_time)
    Ipopt.AddIpoptNumOption(ipopt_prob, "acceptable_obj_change_tol",
                            options.acceptable_obj_change_tol)
    return ipopt_prob, iters, ftrace, xtrace
end

function _hess!(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                obj_factor::Float64, λ::Vector{Float64},
                values::Union{Nothing, Vector{Float64}},
                nparameters::Integer, hess!::Function)
    idx::Int32 = 0
    if values === nothing
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        @inbounds for row in 1:nparameters
            for col in 1:row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        # Symmetric, fill only lower left matrix (values)
        hessian = zeros(Float64, nparameters, nparameters)
        hess!(hessian, x)
        idx = 1
        for row in 1:nparameters
            for col in 1:row
                values[idx] = hessian[row, col] * obj_factor
                idx += 1
            end
        end
    end
    return
end

# In case of of BFGS gradient provide Ipopt with empty hessian struct.
function _hess_empty!(x_arg::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                      obj_factor::Float64, λ::Vector{Float64},
                      values::Union{Nothing, Vector{Float64}})
    return nothing
end

# These function wraps and handles constraints. TODO: Allow optimization under inequality constraints
function eval_jac_g!(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                     values::Union{Nothing, Vector{Float64}})
    return
end
function eval_g!(x::Vector{Float64}, g::Vector{Float64})
    return
end

# Callback for Ipopt allowing number of iterations to be stored.
function intermediate_ipopt(alg_mod::Cint, iter_count::Cint, obj_value::Float64,
                            inf_pr::Float64, inf_du::Float64, mu::Float64, d_norm::Float64,
                            regularization_size::Float64, alpha_du::Float64,
                            alpha_pr::Float64, ls_trials::Cint, iters, ipopt_prob,
                            save_trace::Bool, ftrace::Vector{Float64},
                            xtrace::Vector{Vector{Float64}})
    iters[1] = Int(iter_count)
    if save_trace == true
        push!(ftrace, obj_value)
        push!(xtrace, deepcopy(ipopt_prob.x))
    end
    return true
end

end
