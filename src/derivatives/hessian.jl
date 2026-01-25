function hess!(
        hess::Matrix{T}, x::Vector{T}, _nllh::Function, model_info::ModelInfo,
        cfg::ForwardDiff.HessianConfig
    )::Nothing where {T <: AbstractFloat}
    @unpack simulation_info = model_info

    # If Hessian computation failed a zero Hessian is returned
    simulation_info.could_solve[1] = true
    if _could_solveode_nllh(simulation_info)
        try
            ForwardDiff.hessian!(hess, _nllh, x, cfg)
            @views hess .= Symmetric(hess)
        catch
            fill!(hess, 0.0)
        end
    else
        fill!(hess, 0.0)
    end
    if simulation_info.could_solve[1] != true
        fill!(hess, 0.0)
        return nothing
    end
    return nothing
end

function hess_split!(
        hess::Matrix{T}, x::Vector{T}, _nllh::Function, model_info::ModelInfo;
        cids::Vector{Symbol} = [:all]
    )::Nothing where {T <: AbstractFloat}
    @unpack simulation_info, xindices = model_info

    # If Hessian computation failed a zero Hessian is returned. Here a Hessian is computed
    # for each condition-id, only using parameter present for said condition
    simulation_info.could_solve[1] = true
    fill!(hess, 0.0)
    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        simid = simulation_info.conditionids[:simulation][i]
        ix_simid = _get_ixdynamic_simid(simid, xindices; full_x = true)
        xinput = x[ix_simid]

        hess_tmp = zeros(eltype(x), length(xinput), length(xinput))
        _nllh_cid = (_xinput) -> begin
            _x = convert.(eltype(_xinput), x)
            _x[ix_simid] .= _xinput
            return _nllh(_x, [cid])
        end
        try
            ForwardDiff.hessian!(hess_tmp, _nllh_cid, xinput)
        catch
            fill(hess, 0.0)
            return nothing
        end
        for i in eachindex(ix_simid)
            for j in eachindex(ix_simid)
                hess[ix_simid[i], ix_simid[j]] += hess_tmp[i, j]
            end
        end
    end
    if simulation_info.could_solve[1] != true
        fill!(hess, 0.0)
        return nothing
    end
    return nothing
end

function hess_block!(
        hess::Matrix{T}, x::Vector{T}, _nllh_not_solveode::Function,
        _nllh_solveode::Function, probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
        cfg::ForwardDiff.HessianConfig; cids::Vector{Symbol} = [:all]
    )::Nothing where {T <: AbstractFloat}
    @unpack simulation_info, xindices = model_info
    cache = probinfo.cache

    split_x!(x, xindices, cache; xdynamic_full = true)
    xdynamic_grad = cache.xdynamic_grad

    # If Hessian computation failed a zero Hessian is returned.
    simulation_info.could_solve[1] = true
    fill!(hess, 0.0)
    try
        # Even if xdynamic is empty the ODE must be solved to get the Hessian of the
        # parameters not appearing in the ODE
        if !isempty(xdynamic_grad)
            ix = xindices.indices_est[:est_to_dynamic]
            xdynamic = get_tmp(probinfo.cache.xdynamic, x)
            @views ForwardDiff.hessian!(hess[ix, ix], _nllh_solveode, xdynamic, cfg)
        else
            _nllh_solveode(xdynamic_grad)
        end
    catch
        fill!(hess, 0.0)
        return nothing
    end
    if simulation_info.could_solve[1] != true
        fill!(hess, 0.0)
        return nothing
    end

    ix_not_system = xindices.indices_est[:est_to_not_system]
    x_not_system = @view x[ix_not_system]
    @views ForwardDiff.hessian!(
        hess[ix_not_system, ix_not_system], _nllh_not_solveode, x_not_system
    )
    return nothing
end

function hess_block_split!(
        hess::Matrix{T}, x::Vector{T}, _nllh_not_solveode::Function,
        _nllh_solveode::Function, probinfo::PEtabODEProblemInfo,
        model_info::ModelInfo;
        cids::Vector{Symbol} = [:all]
    )::Nothing where {
        T <:
        AbstractFloat,
    }
    @unpack simulation_info, xindices, priors = model_info
    cache = probinfo.cache
    split_x!(x, xindices, cache; xdynamic_full = true)

    # If Hessian computation failed a zero Hessian is returned. Here a Hessian is computed
    # for each condition-id, only using parameter present for said condition
    simulation_info.could_solve[1] = true
    fill!(hess, 0.0)
    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        simid = simulation_info.conditionids[:simulation][i]
        ixdynamic_simid = _get_ixdynamic_simid(simid, xindices; ml_pre_simulate = true)
        xinput = x[ixdynamic_simid]

        hess_tmp = zeros(eltype(x), length(xinput), length(xinput))
        _nllh_cid = (_xinput) -> begin
            xdynamic = get_tmp(cache.xdynamic, _xinput)
            @views xdynamic[ixdynamic_simid] .= _xinput
            return _nllh_solveode(xdynamic, [cid])
        end
        try
            ForwardDiff.hessian!(hess_tmp, _nllh_cid, xinput)
        catch
            fill!(hess, 0.0)
            return nothing
        end
        for i in eachindex(ixdynamic_simid)
            for j in eachindex(ixdynamic_simid)
                hess[ixdynamic_simid[i], ixdynamic_simid[j]] += hess_tmp[i, j]
            end
        end
    end
    if simulation_info.could_solve[1] != true
        fill!(hess, 0.0)
        return nothing
    end

    ix_not_system = xindices.indices_est[:est_to_not_system]
    x_not_system = @view x[ix_not_system]
    @views ForwardDiff.hessian!(
        hess[ix_not_system, ix_not_system], _nllh_not_solveode, x_not_system
    )
    return nothing
end

function hess_GN!(
        out::Matrix{T}, x::Vector{T}, _residuals_not_solveode::Function,
        _solve_conditions!::Function, probinfo::PEtabODEProblemInfo,
        model_info::ModelInfo, cfg::ForwardDiff.JacobianConfig,
        cfg_not_solve_ode::ForwardDiff.JacobianConfig; ret_jacobian::Bool = false,
        cids::Vector{Symbol} = [:all]
    )::Nothing where {T <: AbstractFloat}
    @unpack xindices = model_info
    cache = probinfo.cache
    @unpack jacobian_gn, residuals_gn = cache

    # See comment in gradient.jl on Jacobian of neural-net
    if probinfo.split_over_conditions == true
        _jac_ml_model_pre_simulate!(probinfo, model_info)
    end

    fill!(out, 0.0)
    fill!(jacobian_gn, 0.0)
    split_x!(x, xindices, cache; xdynamic_full = true)
    _jac = @view jacobian_gn[xindices.indices_est[:est_to_dynamic], :]
    _jac_residuals_xdynamic!(
        _jac, _solve_conditions!, probinfo, model_info, cfg;
        cids = cids
    )
    # Happens when at least one forward pass fails
    if !isempty(cache.xdynamic_grad) && all(_jac .== 0.0)
        return nothing
    end

    x_not_system = @view x[xindices.indices_est[:est_to_not_system]]
    @views ForwardDiff.jacobian!(
        jacobian_gn[xindices.indices_est[:est_to_not_system], :]',
        _residuals_not_solveode, residuals_gn, x_not_system,
        cfg_not_solve_ode
    )

    # In case of testing we might want to return the jacobian, else we are interested
    # in the Guass-Newton approximaiton.
    if ret_jacobian == false
        out .= jacobian_gn * transpose(jacobian_gn)
    else
        out .= jacobian_gn
    end

    # Reset such that neural-nets pre ODE no longer have status of having been evaluated
    _reset_ml_pre_simulate!(probinfo)
    return nothing
end
