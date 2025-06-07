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
                         split_over_conditions::Union{Nothing, Bool} = nothing,
                         reuse_sensitivities::Bool = false,
                         verbose::Bool = false,
                         custom_values::Union{Nothing, Dict} = nothing)::PEtabODEProblem
    _logging(:Build_PEtabODEProblem, verbose; name = model.name)

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

    # Useful functions for getting diagnostics
    _logging(:Build_chi2_res_sim, verbose)
    btime = @elapsed begin
        _chi2 = (x; array = false) -> begin
            _ = nllh(x)
            vals = model_info.petab_measurements.chi2_values
            return array == true ? vals : sum(vals)
        end
        _residuals = (x; array = true) -> begin
            _ = nllh(x)
            vals = model_info.petab_measurements.residuals
            return array == true ? vals : sum(vals)
        end
        _simulated_values = (x) -> begin
            _ = nllh(x)
            return model_info.petab_measurements.simulated_values
        end
    end
    _logging(:Build_chi2_res_sim, verbose; time = btime)

    # Relevant information for the unknown model parameters
    xnames = model_info.xindices.xids[:estimate]
    xnames_ps = model_info.xindices.xids[:estimate_ps]
    nestimate = length(xnames)
    lb = _get_bounds(model_info, xnames, xnames_ps, :lower)
    ub = _get_bounds(model_info, xnames, xnames_ps, :upper)
    xnominal = _get_xnominal(model_info, xnames, xnames_ps, false)
    xnominal_transformed = _get_xnominal(model_info, xnames, xnames_ps, true)

    return PEtabODEProblem(nllh, _chi2, grad!, grad, hess!, hess, FIM!, FIM, nllh_grad,
                           prior, grad_prior, hess_prior, _simulated_values, _residuals,
                           probinfo, model_info, nestimate, xnames, xnominal,
                           xnominal_transformed, lb, ub)
end

function _get_prior(model_info::ModelInfo)::Tuple{Function, Function, Function}
    _prior = let minfo = model_info
        @unpack xindices, priors = minfo
        (x) -> begin
            xnames = xindices.xids[:estimate]
            return prior(x, xnames, priors, xindices)
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
            _test_ordering(x, minfo.xindices.xids[:estimate_ps])
            _x = x |> collect
            nllh_val = nllh(_x, pinfo, minfo, [:all], false, res)
            if prior == true && res == false
                # nllh -> negative prior
                return nllh_val - _prior(_x)
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
            _x = x |> collect
            _g = similar(_x)
            _grad_nllh!(_g, _x; isremade = isremade)
            if prior
                # nllh -> negative prior
                _g .+= grad_prior(_x) .* -1
            end
            g .= _g
            return nothing
        end
    end
    _grad = let _grad! = _grad!
        (x; prior = true, isremade = false) -> begin
            gradient = similar(x)
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
    @unpack xdynamic_mech = cache
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
            _x = x |> collect
            _H = H |> collect
            if hessian_method == :GassNewton
                _hess_nllh!(_H, _x; isremade = isremade)
            else
                _hess_nllh!(_H, _x)
            end
            if prior && ret_jacobian == false
                # nllh -> negative prior
                _H .+= hess_prior(_x) .* -1
            end
            H .= _H
            return nothing
        end
    end
    _hess = (x; prior = true) -> begin
        if hessian_method == :GaussNewton && ret_jacobian == true
            H = zeros(eltype(x), length(x), length(model_info.petab_measurements.time))
        else
            H = zeros(eltype(x), length(x), length(x))
        end
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
        _x = x |> collect
        g = grad(x; prior = prior)
        x_notode = @view _x[model_info.xindices.xindices[:not_system_mech]]
        nllh = _nllh_not_solveode(x_notode)
        if prior
            nllh += _prior(_x)
        end
        return nllh, g
    end
    return _nllh_grad
end

function _get_bounds(model_info::ModelInfo, xnames::Vector{Symbol}, xnames_ps::Vector{Symbol}, which::Symbol)
    @unpack petab_parameters, petab_net_parameters, xindices = model_info

    # Mechanistic parameters has its bounds like a Vector
    ix_mech = _get_ixnames_mech(xnames, petab_parameters)
    xnames_mech = xnames[ix_mech]
    ix = [findfirst(x -> x == id, petab_parameters.parameter_id) for id in xnames_mech]
    if which == :lower
        bounds = petab_parameters.lower_bounds[ix]
    else
        bounds = petab_parameters.upper_bounds[ix]
    end
    transform_x!(bounds, xnames_mech, xindices, to_xscale = true)
    xmech_bounds = NamedTuple(xnames_ps[ix_mech] .=> bounds)

    # Each network has its bounds as a ComponentArray
    xnames_nn = xnames[setdiff(1:length(xnames), ix_mech)]
    bounds = Vector{ComponentArray}(undef, length(xnames_nn))
    for (i, ml_model_id) in pairs(xnames_nn)
        ml_model = model_info.model.ml_models[ml_model_id]
        bounds[i] = _get_ml_model_initialparameters(ml_model)
        if which == :lower
            bounds[i] .= -Inf
        else
            bounds[i] .= Inf
        end
    end
    xnn_bounds = (xnames_nn .=> bounds) |> NamedTuple
    return merge(xmech_bounds, xnn_bounds) |> ComponentArray
end

function _get_xnominal(model_info::ModelInfo, xnames::Vector{Symbol},
                       xnames_ps::Vector{Symbol}, transform::Bool)
    @unpack petab_parameters, xindices, model = model_info
    @unpack ml_models, paths, petab_tables = model

    # Mechanistic parameters which are returned as a Vector
    ix_mech = _get_ixnames_mech(xnames, petab_parameters)
    xnominal_mech = _get_xnominal_mech(xnames[ix_mech], petab_parameters)
    if transform == true
        transform_x!(xnominal_mech, xnames[ix_mech], xindices, to_xscale = true)
        xmech = (xnames_ps[ix_mech] .=> xnominal_mech) |> NamedTuple
    else
        xmech = (xnames[ix_mech] .=> xnominal_mech) |> NamedTuple
    end

    # Each network has its parameters as a ComponentArray
    xnames_nn = xnames[setdiff(1:length(xnames), ix_mech)]
    xnominal_nn = Vector{ComponentArray}(undef, length(xnames_nn))
    for (i, ml_model_id) in pairs(xnames_nn)
        ml_model = model_info.model.ml_models[ml_model_id]
        psnet = _get_ml_model_initialparameters(ml_model)
        set_ml_model_ps!(psnet, ml_model_id, ml_models, paths, petab_tables)
        xnominal_nn[i] = psnet
    end
    xnn = (xnames_nn .=> xnominal_nn) |> NamedTuple
    return ComponentArray(merge(xmech, xnn))
end

function _get_ixnames_mech(xnames::Vector{Symbol}, petab_parameters::PEtabParameters)::Vector{Int64}
    return findall(x -> x in petab_parameters.parameter_id, xnames)
end

function _get_xnominal_mech(xnames_mech::Vector{Symbol}, petab_parameters::PEtabParameters)::Vector{Float64}
    ix = [findfirst(x -> x == id, petab_parameters.parameter_id) for id in xnames_mech]
    return petab_parameters.nominal_value[ix]
end

function _get_xnames(petab_parameters::PEtabMLParameters)::Vector{Symbol}
    ix = findall(x -> x == true, petab_parameters.estimate)
    return petab_parameters.parameter_id[ix]
end

function _test_ordering(x::ComponentArray, xnames_ps::Vector{Symbol})::Nothing
    if !all(propertynames(x) .== xnames_ps)
        throw(PEtabInputError("Input ComponentArray x to the PEtab nllh function \
                               has wrong ordering or parameter names. In x the \
                               parameters must appear in the order of $xnames_ps"))
    end
    return nothing
end
_test_ordering(x::AbstractVector, names_ps::Vector{Symbol})::Nothing = nothing
