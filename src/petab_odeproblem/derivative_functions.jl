#=
    Gradient functions
=#

function _get_grad_forward_AD(probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, gradient_method, chunksize = probinfo
    @unpack sensealg, cache = probinfo
    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_AD = true, cids = cids)

    if split_over_conditions == false
        _nllh_solveode = _get_nllh_solveode(probinfo, model_info; grad_xdynamic = true)

        @unpack xdynamic = cache
        chunksize_use = _get_chunksize(chunksize, xdynamic)
        cfg = ForwardDiff.GradientConfig(_nllh_solveode, xdynamic, chunksize_use)
        _grad! = let _nllh_not_solveode = _nllh_not_solveode,
            _nllh_solveode = _nllh_solveode, cfg = cfg, minfo = model_info, pinfo = probinfo

            (grad, x; isremade = false) -> grad_forward_AD!(grad, x, _nllh_not_solveode,
                                                            _nllh_solveode, cfg, pinfo,
                                                            minfo; isremade = isremade)
        end
    end

    if split_over_conditions == true
        _nllh_solveode = _get_nllh_solveode(probinfo, model_info; cid = true,
                                            grad_xdynamic = true)

        _grad! = let _nllh_not_solveode = _nllh_not_solveode,
            _nllh_solveode = _nllh_solveode, minfo = model_info, pinfo = probinfo

            (g, x; isremade = false) -> grad_forward_AD_split!(g, x, _nllh_not_solveode,
                                                               _nllh_solveode, pinfo, minfo)
        end
    end
    return _grad!
end

function _get_grad_forward_eqs(probinfo::PEtabODEProblemInfo,
                               model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, gradient_method, chunksize = probinfo
    @unpack sensealg, cache = probinfo
    @unpack xdynamic, odesols = cache
    chunksize_use = _get_chunksize(chunksize, xdynamic)

    if sensealg == :ForwardDiff && split_over_conditions == false
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x) -> solve_conditions!(sols, x, pinfo, minfo; sensitivites_AD = true)
        end
        cfg = ForwardDiff.JacobianConfig(_solve_conditions!, odesols, xdynamic,
                                         chunksize_use)
    end

    if sensealg == :ForwardDiff && split_over_conditions == true
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x, cid) -> solve_conditions!(sols, x, pinfo, minfo; cids = cid,
                                                sensitivites_AD = true)
        end
        cfg = ForwardDiff.JacobianConfig(_solve_conditions!, odesols, xdynamic,
                                         chunksize_use)
    end

    if sensealg != :ForwardDiff
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (x, cid) -> solve_conditions!(minfo, x, pinfo; cids = cid, sensitivites = true)
        end
        cfg = nothing
    end

    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_eqs = true, cids = cids)

    _grad! = let _nllh_not_solveode = _nllh_not_solveode,
        _solve_conditions! = _solve_conditions!, minfo = model_info, pinfo = probinfo,
        cfg = cfg, Cids = cids

        (g, x; isremade = false, cids = Cids) -> grad_forward_eqs!(g, x, _nllh_not_solveode,
                                                      _solve_conditions!, pinfo, minfo,
                                                      cfg; cids = cids,
                                                      isremade = isremade)
    end
    return _grad!
end

#=
    Hessian functions
=#

function _get_hess_forward_AD(probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, chunksize = probinfo
    if split_over_conditions == false
        _nllh = let pinfo = probinfo, minfo = model_info, Cids = cids
            (x) -> nllh(x, pinfo, minfo, Cids, true, false)
        end
        nestimate = length(model_info.xindices.xids[:estimate])
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
                                    model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, chunksize = probinfo
    xdynamic = probinfo.cache.xdynamic

    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_AD = true, cids = cids)

    if split_over_conditions == false
        _nllh_solveode = let pinfo = probinfo, minfo = model_info, Cids = cids
            @unpack xnoise, xobservable, xnondynamic = pinfo.cache
            (x) -> nllh_solveode(x, xnoise, xobservable, xnondynamic, pinfo, minfo;
                                 grad_xdynamic = true, cids = Cids)
        end

        chunksize_use = _get_chunksize(chunksize, xdynamic)
        cfg = ForwardDiff.HessianConfig(_nllh_solveode, xdynamic, chunksize_use)
        _hess_nllh! = let _nllh_solveode = _nllh_solveode,
            _nllh_not_solveode = _nllh_not_solveode, pinfo = probinfo, minfo = model_info, Cids = cids,
            cfg = cfg

            (H, x) -> hess_block!(H, x, _nllh_not_solveode, _nllh_solveode, pinfo,
                                  minfo, cfg; cids = Cids)
        end
    end

    if split_over_conditions == true
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            @unpack xnoise, xobservable, xnondynamic = pinfo.cache
            (x, cid) -> nllh_solveode(x, xnoise, xobservable, xnondynamic,
                                      pinfo, minfo,
                                      grad_xdynamic = true,
                                      cids = cid)
        end

        _hess_nllh! = let _nllh_solveode = _nllh_solveode,
            _nllh_not_solveode = _nllh_not_solveode, pinfo = probinfo, minfo = model_info, Cids = cids

            (H, x) -> hess_block_split!(H, x, _nllh_not_solveode, _nllh_solveode,
                                        pinfo, minfo; cids = Cids)
        end
    end
    return _hess_nllh!
end

function _get_hess_gaussnewton(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                               ret_jacobian::Bool; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, chunksize, cache = probinfo
    xdynamic = probinfo.cache.xdynamic

    if split_over_conditions == false
        _solve_conditions! = let pinfo = probinfo, minfo = model_info, Cids = cids
            (sols, x) -> solve_conditions!(sols, x, pinfo, minfo; cids = Cids,
                                           sensitivites_AD = true)
        end
    end
    if split_over_conditions == true
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x, cid) -> solve_conditions!(sols, x, pinfo, minfo; cids = cid,
                                                sensitivites_AD = true)
        end
    end
    chunksize_use = _get_chunksize(chunksize, xdynamic)
    cfg = cfg = ForwardDiff.JacobianConfig(_solve_conditions!,
                                           cache.odesols,
                                           cache.xdynamic, chunksize_use)

    _residuals_not_solveode = let pinfo = probinfo, minfo = model_info, Cids = cids
        ixnoise = minfo.xindices.xindices_notsys[:noise]
        ixobservable = minfo.xindices.xindices_notsys[:observable]
        ixnondynamic = minfo.xindices.xindices_notsys[:nondynamic]
        (residuals, x) -> begin
            residuals_not_solveode(residuals, x[ixnoise], x[ixobservable],
                                   x[ixnondynamic], pinfo, minfo; cids = Cids)
        end
    end

    xnot_ode = zeros(Float64, length(model_info.xindices.xids[:not_system]))
    cfg_notsolve = ForwardDiff.JacobianConfig(_residuals_not_solveode,
                                              cache.residuals_gn, xnot_ode,
                                              ForwardDiff.Chunk(xnot_ode))
    _hess_nllh! = let _residuals_not_solveode = _residuals_not_solveode,
        pinfo = probinfo, minfo = model_info, cfg = cfg, cfg_notsolve = cfg_notsolve,
        ret_jacobian = ret_jacobian, _solve_conditions! = _solve_conditions!, Cids = cids

        (H, x; isremade = false) -> hess_GN!(H, x, _residuals_not_solveode,
                                             _solve_conditions!, pinfo, minfo, cfg,
                                             cfg_notsolve; cids = Cids,
                                             isremade = isremade,
                                             ret_jacobian = ret_jacobian)
    end
    return _hess_nllh!
end

#=
    Helpers
=#

function _get_nllh_not_solveode(probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                                grad_forward_AD::Bool = false, grad_adjoint::Bool = false,
                                grad_forward_eqs::Bool = false, cids::AbstractVector{Symbol}=[:all])::Function
    _nllh_not_solveode = let pinfo = probinfo, minfo = model_info, Cids = cids
        ixnoise = minfo.xindices.xindices_notsys[:noise]
        ixobservable = minfo.xindices.xindices_notsys[:observable]
        ixnondynamic = minfo.xindices.xindices_notsys[:nondynamic]
        (x) -> nllh_not_solveode(x[ixnoise], x[ixobservable], x[ixnondynamic], pinfo, minfo;
                                 cids = Cids, grad_forward_AD = grad_forward_AD,
                                 grad_forward_eqs = grad_forward_eqs,
                                 grad_adjoint = grad_adjoint)
    end
    return _nllh_not_solveode
end

function _get_nllh_solveode(probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                            grad_xdynamic::Bool = false, cid::Bool = false)
    if cid == false
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            @unpack xnoise, xobservable, xnondynamic = pinfo.cache
            (x) -> nllh_solveode(x, xnoise, xobservable, xnondynamic, pinfo, minfo;
                                 grad_xdynamic = grad_xdynamic, cids = [:all])
        end
    else
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            @unpack xnoise, xobservable, xnondynamic = pinfo.cache
            (x, _cid) -> nllh_solveode(x, xnoise, xobservable, xnondynamic, pinfo, minfo;
                                       grad_xdynamic = grad_xdynamic, cids = _cid)
        end
    end
    return _nllh_solveode
end
