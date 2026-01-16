#=
    Gradient functions
=#

function _get_grad_forward_AD(probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo)::Function
    @unpack split_over_conditions, gradient_method, chunksize = probinfo
    @unpack sensealg, cache = probinfo
    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_AD = true)

    if split_over_conditions == false
        _nllh_solveode = _get_nllh_solveode(probinfo, model_info; grad_xdynamic = true)

        @unpack xdynamic_grad = cache
        chunksize_use = _get_chunksize(chunksize, xdynamic_grad)
        cfg = ForwardDiff.GradientConfig(_nllh_solveode, xdynamic_grad, chunksize_use)
        _grad! = let _nllh_not_solveode = _nllh_not_solveode,
            _nllh_solveode = _nllh_solveode, cfg = cfg, minfo = model_info, pinfo = probinfo

            (grad, x) -> grad_forward_AD!(grad, x, _nllh_not_solveode, _nllh_solveode, cfg,
                                          pinfo, minfo)
        end
    end

    if split_over_conditions == true
        _nllh_solveode = _get_nllh_solveode(probinfo, model_info; cid = true,
                                            grad_xdynamic = true)

        _grad! = let _nllh_not_solveode = _nllh_not_solveode,
            _nllh_solveode = _nllh_solveode, minfo = model_info, pinfo = probinfo

            (g, x) -> grad_forward_AD_split!(g, x, _nllh_not_solveode, _nllh_solveode,
                                             pinfo, minfo)
        end
    end
    return _grad!
end

function _get_grad_forward_eqs(
        probinfo::PEtabODEProblemInfo, model_info::ModelInfo
    )::Function
    @unpack split_over_conditions, gradient_method, chunksize = probinfo
    @unpack sensealg, cache = probinfo
    @unpack xdynamic_grad, odesols = cache
    chunksize_use = _get_chunksize(chunksize, xdynamic_grad)

    if sensealg == :ForwardDiff && split_over_conditions == false
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x) -> solve_conditions!(sols, x, pinfo, minfo; sensitivities_AD = true)
        end
        cfg = ForwardDiff.JacobianConfig(_solve_conditions!, odesols, xdynamic_grad,
                                         chunksize_use)
    end

    if sensealg == :ForwardDiff && split_over_conditions == true
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x, cid) -> solve_conditions!(sols, x, pinfo, minfo; cids = cid,
                                                sensitivities_AD = true)
        end
        cfg = ForwardDiff.JacobianConfig(_solve_conditions!, odesols, xdynamic_grad,
                                         chunksize_use)
    end

    if sensealg != :ForwardDiff
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (x, cid) -> begin
                xdynamic_mech, xnn = split_xdynamic(x, minfo.xindices, pinfo.cache)
                return solve_conditions!(
                    minfo, xdynamic_mech, xnn, pinfo; cids = cid, sensitivities = true
                )
            end
        end
        cfg = nothing
    end

    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_eqs = true)

    _grad! = let _nllh_not_solveode = _nllh_not_solveode,
        _solve_conditions! = _solve_conditions!, minfo = model_info, pinfo = probinfo,
        cfg = cfg

        (g, x) -> grad_forward_eqs!(g, x, _nllh_not_solveode, _solve_conditions!, pinfo,
                                    minfo, cfg; cids = [:all])
    end
    return _grad!
end

#=
    Hessian functions
=#

function _get_hess_forward_AD(probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo)::Function
    @unpack split_over_conditions, chunksize = probinfo
    if split_over_conditions == false
        _nllh = let pinfo = probinfo, minfo = model_info
            (x) -> nllh(x, pinfo, minfo, [:all], true, false)
        end
        nestimate = _get_nx_estimate(model_info)
        chunksize_use = _get_chunksize(chunksize, zeros(nestimate))
        cfg = ForwardDiff.HessianConfig(_nllh, zeros(nestimate), chunksize_use)
        _hess_nllh! = let _nllh = _nllh, cfg = cfg, minfo = model_info
            (H, x) -> hess!(H, x, _nllh, minfo, cfg)
        end
    end

    if split_over_conditions == true
        _nllh = let pinfo = probinfo, minfo = model_info
            (x, cid) -> nllh(x, pinfo, minfo, cid, true, false)
        end
        _hess_nllh! = let _nllh = _nllh, minfo = model_info
            (H, x) -> hess_split!(H, x, _nllh, minfo)
        end
    end
    return _hess_nllh!
end

function _get_hess_block_forward_AD(probinfo::PEtabODEProblemInfo,
                                    model_info::ModelInfo)::Function
    @unpack split_over_conditions, chunksize = probinfo
    xdynamic_grad = probinfo.cache.xdynamic_grad

    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_AD = true)

    if split_over_conditions == false
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            xnoise, xobservable, xnondynamic_mech = _get_x_notsystem(pinfo.cache, 1.0)
            (x) -> begin
                xmech, xnn = split_xdynamic(x, minfo.xindices, probinfo.cache)
                return nllh_solveode(xmech, xnoise, xobservable, xnondynamic_mech, xnn, pinfo,
                                     minfo; grad_xdynamic = true, cids = [:all])
            end
        end
        chunksize_use = _get_chunksize(chunksize, xdynamic_grad)
        cfg = ForwardDiff.HessianConfig(_nllh_solveode, xdynamic_grad, chunksize_use)
        _hess_nllh! = let _nllh_solveode = _nllh_solveode,
            _nllh_not_solveode = _nllh_not_solveode, pinfo = probinfo, minfo = model_info,
            cfg = cfg
            (H, x) -> hess_block!(H, x, _nllh_not_solveode, _nllh_solveode, pinfo,
                                  minfo, cfg; cids = [:all])
        end
    end

    if split_over_conditions == true
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            xnoise, xobservable, xnondynamic_mech = _get_x_notsystem(pinfo.cache, 1.0)
            (x, cid) -> begin
                xmech, xnn = split_xdynamic(x, minfo.xindices, probinfo.cache)
                return nllh_solveode(xmech, xnoise, xobservable, xnondynamic_mech, xnn,
                                     pinfo, minfo, grad_xdynamic = true, cids = cid)
            end
        end

        _hess_nllh! = let _nllh_solveode = _nllh_solveode,
            _nllh_not_solveode = _nllh_not_solveode, pinfo = probinfo, minfo = model_info

            (H, x) -> hess_block_split!(H, x, _nllh_not_solveode, _nllh_solveode,
                                        pinfo, minfo; cids = [:all])
        end
    end
    return _hess_nllh!
end

function _get_hess_gaussnewton(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                               ret_jacobian::Bool)::Function
    @unpack split_over_conditions, chunksize, cache = probinfo

    if split_over_conditions == false
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x) -> solve_conditions!(sols, x, pinfo, minfo; cids = [:all],
                                           sensitivities_AD = true)
        end
    end
    if split_over_conditions == true
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x, cid) -> solve_conditions!(sols, x, pinfo, minfo; cids = cid,
                                                sensitivities_AD = true)
        end
    end
    chunksize_use = _get_chunksize(chunksize, cache.xdynamic_grad)
    cfg = cfg = ForwardDiff.JacobianConfig(_solve_conditions!, cache.odesols,
                                           cache.xdynamic_grad, chunksize_use)

    _residuals_not_solveode = let pinfo = probinfo, minfo = model_info
        (residuals, x) -> begin
            xn, xo, xnm, xnn = split_x_notsystem(x, minfo.xindices, pinfo.cache)
            residuals_not_solveode(residuals, xn, xo, xnm, xnn, pinfo, minfo; cids = [:all])
        end
    end

    xnot_ode = zeros(Float64, length(model_info.xindices.xindices[:not_system_tot]))
    cfg_notsolve = ForwardDiff.JacobianConfig(_residuals_not_solveode,
                                              cache.residuals_gn, xnot_ode,
                                              ForwardDiff.Chunk(xnot_ode))
    _hess_nllh! = let _residuals_not_solveode = _residuals_not_solveode,
        pinfo = probinfo, minfo = model_info, cfg = cfg, cfg_notsolve = cfg_notsolve,
        ret_jacobian = ret_jacobian, _solve_conditions! = _solve_conditions!

        (H, x) -> hess_GN!(H, x, _residuals_not_solveode, _solve_conditions!, pinfo, minfo,
                           cfg, cfg_notsolve; cids = [:all], ret_jacobian = ret_jacobian)
    end
    return _hess_nllh!
end

#=
    Helpers
=#

function _get_nllh_not_solveode(probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                                grad_forward_AD::Bool = false, grad_adjoint::Bool = false,
                                grad_forward_eqs::Bool = false)::Function
    _nllh_not_solveode = let pinfo = probinfo, minfo = model_info
        (x) -> begin
            xn, xo, xnm, xnn = split_x_notsystem(x, minfo.xindices, pinfo.cache)
            return nllh_not_solveode(xn, xo, xnm, xnn, pinfo, minfo; cids = [:all],
                                     grad_forward_AD = grad_forward_AD,
                                     grad_forward_eqs = grad_forward_eqs,
                                     grad_adjoint = grad_adjoint)
        end
    end
    return _nllh_not_solveode
end

function _get_nllh_solveode(probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                            grad_xdynamic::Bool = false, cid::Bool = false)::Function
    if cid == false
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            xnoise, xobservable, xnondynamic_mech = _get_x_notsystem(pinfo.cache, 1.0)
            (x) -> begin
                xmech, xnn = split_xdynamic(x, minfo.xindices, probinfo.cache)
                return nllh_solveode(xmech, xnoise, xobservable, xnondynamic_mech, xnn, pinfo, minfo; grad_xdynamic = grad_xdynamic, cids = [:all])
            end
        end
    else
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            xnoise, xobservable, xnondynamic_mech = _get_x_notsystem(pinfo.cache, 1.0)
            (x, cid) -> begin
                xmech, xnn = split_xdynamic(x, minfo.xindices, probinfo.cache)
                return nllh_solveode(xmech, xnoise, xobservable, xnondynamic_mech, xnn, pinfo,
                                     minfo; grad_xdynamic = grad_xdynamic, cids = cid)
            end
        end
    end
    return _nllh_solveode
end

function _get_x_notsystem(cache::PEtabODEProblemCache, x::T)::NTuple{3, AbstractVector{T}} where T<:Real
    xnoise = get_tmp(cache.xnoise, x)
    xobservable = get_tmp(cache.xobservable, x)
    xnondynamic_mech = get_tmp(cache.xnondynamic_mech, x)
    return xnoise, xobservable, xnondynamic_mech
end

function _get_x_not_nn(cache::PEtabODEProblemCache, x::T)::NTuple{4, AbstractVector{T}} where T<:Real
    xnoise, xobservable, xnondynamic_mech = _get_x_notsystem(cache, x)
    xdynamic = get_tmp(cache.xdynamic_mech, x)
    return xnoise, xobservable, xnondynamic_mech, xdynamic
end

function _get_nx_estimate(model_info::ModelInfo)::Int64
    nestimate = length(model_info.xindices.xindices[:not_system_tot]) +
                length(model_info.xindices.xindices_dynamic[:xest_to_xdynamic])
    return nestimate
end

function split_x_notsystem(x, xindices::ParameterIndices, cache::PEtabODEProblemCache)
    xnoise = get_tmp(cache.xnoise, x)
    xobservable = get_tmp(cache.xobservable, x)
    xnondynamic_mech = get_tmp(cache.xnondynamic_mech, x)
    xnoise .= @view x[xindices.xindices_notsys[:noise]]
    xobservable .= @view x[xindices.xindices_notsys[:observable]]
    xnondynamic_mech .= @view x[xindices.xindices_notsys[:nondynamic_mech]]
    for ml_id in xindices.xids[:ml_nondynamic]
        !(ml_id in xindices.xids[:ml_est]) && continue
        xnn = get_tmp(cache.xnn[ml_id], x)
        xnn .= x[xindices.xindices_notsys[ml_id]]
        cache.xnn_dict[ml_id] = xnn
    end
    return xnoise, xobservable, xnondynamic_mech, cache.xnn_dict
end
