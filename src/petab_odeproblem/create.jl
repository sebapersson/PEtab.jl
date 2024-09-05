function PEtabODEProblem(model::PEtabModel;
                         odesolver::Union{Nothing, ODESolver} = nothing,
                         odesolver_gradient::Union{Nothing, ODESolver} = nothing,
                         ss_solver::Union{Nothing, SteadyStateSolver} = nothing,
                         ss_solver_gradient::Union{Nothing, SteadyStateSolver} = nothing,
                         gradient_method::Union{Nothing, Symbol} = nothing,
                         hessian_method::Union{Nothing, Symbol} = nothing,
                         FIM_method::Union{Nothing, Symbol} = nothing,
                         sparse_jacobian::Union{Nothing, Bool} = nothing,
                         specialize_level = SciMLBase.FullSpecialize,
                         sensealg = nothing,
                         sensealg_ss = nothing,
                         chunksize::Union{Nothing, Int64} = nothing,
                         split_over_conditions::Bool = false,
                         reuse_sensitivities::Bool = false,
                         verbose::Bool = true,
                         custom_values::Union{Nothing, Dict} = nothing)::PEtabODEProblem
    _logging(:Build_PEtabODEProblem, verbose; name = model.modelname)

    # To bookeep parameters, measurements, etc...
    model_info = ModelInfo(model, sensealg, custom_values)
    # All ODE-relevent info for the problem, e.g. solvers, gradient method ...
    probinfo = PEtabODEProblemInfo(model, model_info, odesolver, odesolver_gradient,
                                   ss_solver, ss_solver_gradient, gradient_method,
                                   hessian_method, FIM_method, sensealg, sensealg_ss,
                                   reuse_sensitivities, sparse_jacobian, specialize_level,
                                   chunksize, split_over_conditions, verbose)

    # The prior enters into the nllh, grad, and hessian functions and is evaluated by
    # default (keyword user can toggle). Grad and hess are not inplace, in order to
    # to not overwrite the nllh hess/grad when evaluating total grad/hess
    prior, grad_prior, hess_prior = _get_prior(model_info)

    _logging(:Build_nllh, verbose)
    btime = @elapsed begin
        nllh = _get_nllh(probinfo, model_info, prior, false)
    end
    _logging(:Build_nllh, verbose; time = btime)

    _logging(:Build_gradient, verbose; method = probinfo.gradient_method)
    btime = @elapsed begin
        method = probinfo.gradient_method
        grad!, grad = _get_grad(Val(method), probinfo, model_info, grad_prior)
        nllh_grad = _get_nllh_grad(method, grad, prior, probinfo, model_info)
    end
    _logging(:Build_gradient, verbose; time = btime)

    _logging(:Build_hessian, verbose; method = probinfo.hessian_method)
    btime = @elapsed begin
        hess!, hess = _get_hess(probinfo, model_info, hess_prior)
        FIM!, FIM = _get_hess(probinfo, model_info, hess_prior; FIM = true)
    end
    _logging(:Build_hessian, verbose; time = btime)

    # TODO: Must refactor later
    compute_chi2 = (θ; as_array = false) -> begin
        _ = nllh(θ)
        if as_array == false
            return sum(measurement_info.chi2_values)
        else
            return measurement_info.chi2_values
        end
    end
    compute_residuals = (θ; as_array = false) -> begin
        _ = nllh(θ)
        if as_array == false
            return sum(measurement_info.residuals)
        else
            return measurement_info.residuals
        end
    end
    compute_simulated_values = (θ) -> begin
        _ = nllh(θ)
        return measurement_info.simulated_values
    end

    # Relevant information for the unknown model parameters
    xnames = model_info.θ_indices.xids[:estimate]
    nestimate = length(xnames)
    lb = _get_bounds(model_info, xnames, :lower)
    ub = _get_bounds(model_info, xnames, :upper)
    xnominal = _get_xnominal(model_info, xnames, false)
    xnominal_transformed = _get_xnominal(model_info, xnames, true)

    return PEtabODEProblem(nllh, compute_chi2, grad!, grad, hess!, hess, FIM!, FIM,
                           nllh_grad, prior, grad_prior, hess_prior,
                           compute_simulated_values, compute_residuals, probinfo,
                           model_info, nestimate, xnames, xnominal, xnominal_transformed,
                           lb, ub)
end

function _get_prior(model_info::ModelInfo)::Tuple{Function, Function, Function}
    _prior = let minfo = model_info
        @unpack θ_indices, prior_info = minfo
        (x) -> begin
            xnames = θ_indices.xids[:estimate]
            return prior(x, xnames, prior_info, θ_indices)
        end
    end
    _grad_prior = let p = _prior
        x -> ForwardDiff.gradient(p, x)
    end
    _hess_prior = let p = _prior
        x -> ForwardDiff.hessian(p, x)
    end
    return _prior, _grad_prior, _hess_prior
end

function _get_nllh(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   prior::Function, residuals::Bool)::Function
    _nllh = let pinfo = probinfo, minfo = model_info, res = residuals, _prior = prior
        (x; prior = true) -> begin
            nllh_val = nllh(x, pinfo, minfo, [:all], false, res)
            if prior == true && res == false
                # nllh -> negative prior
                return nllh_val - _prior(x)
            else
                return nllh_val
            end
        end
    end
    return _nllh
end

function _get_grad(method, probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   grad_prior::Function)::Tuple{Function, Function}
    if probinfo.gradient_method == :ForwardDiff
        _grad_nllh! = _get_grad_forward_AD(probinfo, model_info)
    end
    if probinfo.gradient_method == :ForwardEquations
        _grad_nllh! = _get_grad_forward_eqs(probinfo, model_info)
    end

    _grad! = let _grad_nllh! = _grad_nllh!, grad_prior = grad_prior
        (g, x; prior = true, isremade = false) -> begin
            _grad_nllh!(g, x; isremade = isremade)
            if prior
                # nllh -> negative prior
                g .+= grad_prior(x) .* -1
            end
            return nothing
        end
    end
    _grad = let _grad! = _grad!
        (x; prior = true, isremade = isremade) -> begin
            gradient = zeros(Float64, length(x))
            _grad!(gradient, x; prior = prior, isremade = isremade)
            return gradient
        end
    end
    return _grad!, _grad
end

function _get_hess(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                     hess_prior::Function; ret_jacobian::Bool = false,
                     FIM::Bool = false)::Tuple{Function, Function}
    @unpack hessian_method, split_over_conditions, chunksize, cache = probinfo
    @unpack xdynamic = cache
    if FIM == true
        hessian_method = probinfo.FIM_method
    end

    if hessian_method === :ForwardDiff
        _hess_nllh! = _get_hess_forward_AD(probinfo, model_info)
    elseif hessian_method === :BlockForwardDiff
        _hess_nllh! = _get_hess_block_forward_AD(probinfo, model_info)
    elseif hessian_method == :GaussNewton
        _hess_nllh! = _get_hess_gaussnewton(probinfo, model_info, ret_jacobian)
    end

    _hess! = let _hess_nllh! = _hess_nllh!, hess_prior = hess_prior
        (H, x; prior = true, isremade = false) -> begin
            if hessian_method == :GassNewton
                _hess_nllh!(H, x; isremade = isremade)
            else
                _hess_nllh!(H, x)
            end
            if prior
                # nllh -> negative prior
                H .+= hess_prior(x) .* -1
            end
            return nothing
        end
    end
    _hess = (x; prior = true) -> begin
        H = zeros(eltype(x), length(x), length(x))
        _hess!(H, x; prior = prior)
        return H
    end
    return _hess!, _hess
end

function _get_nllh_grad(gradient_method::Symbol, grad::Function, _prior::Function,
                        probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Function
    grad_forward_AD = gradient_method == :ForwardDiff
    grad_forward_eqs = gradient_method == :ForwardEquations
    grad_adjoint = gradient_method == :Adjoint

    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_AD = grad_forward_AD,
                                                grad_forward_eqs = grad_forward_eqs,
                                                grad_adjoint = grad_adjoint)
    _nllh_grad = (x; prior = true) -> begin
        g = grad(x; prior = prior)
        nllh = _nllh_not_solveode(x)
        if prior
            nllh += _prior(x)
        end
        return nllh, g
    end
    return _nllh_grad
end

function _get_bounds(model_info::ModelInfo, xnames::Vector{Symbol},
                     which::Symbol)::Vector{Float64}
    @unpack parameter_info, θ_indices = model_info
    ix = [findfirst(x -> x == id, parameter_info.parameter_id) for id in xnames]
    if which == :lower
        bounds = parameter_info.lower_bounds[ix]
    else
        bounds = parameter_info.upper_bounds[ix]
    end
    transform_x!(bounds, xnames, θ_indices, reverse_transform = true)
    return bounds
end

function _get_xnominal(model_info::ModelInfo, xnames::Vector{Symbol},
                       transform::Bool)::Vector{Float64}
    @unpack parameter_info, θ_indices = model_info
    ix = [findfirst(x -> x == id, parameter_info.parameter_id) for id in xnames]
    xnominal = parameter_info.nominal_value[ix]
    if transform == true
        transform_x!(xnominal, xnames, θ_indices, reverse_transform = true)
    end
    return xnominal
end
