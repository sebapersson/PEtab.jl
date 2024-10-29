function grad_forward_AD!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                          _nllh_solveode::Function, cfg::ForwardDiff.GradientConfig,
                          probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                          cids::Vector{Symbol} = [:all],
                          isremade::Bool = false)::Nothing where {T <: AbstractFloat}
    cache = probinfo.cache
    @unpack simulation_info, xindices, priors = model_info
    @unpack xdynamic_grad, xnotode_grad, nxdynamic = cache

    # As a subset of ForwardDiff chunks might fail, return code status is checked via
    # simulation info
    simulation_info.could_solve[1] = true

    # When remaking the problem order of parameters a subset of xdynamic is fixed which
    # must be accounted for in gradient computations
    fill!(grad, 0.0)
    split_x!(x, xindices, cache; xdynamic_tot = true)
    xdynamic = get_tmp(cache.xdynamic_tot, x)
    if isremade == false || length(xdynamic) == nxdynamic[1]
        tmp = nxdynamic[1]
        nxdynamic[1] = length(xdynamic)
        try
            # In case of no length(xdynamic) = 0 the ODE must still be solved to get
            # the gradient of nondynamic parameters
            if length(xdynamic_grad) != 0
                ForwardDiff.gradient!(xdynamic_grad, _nllh_solveode, xdynamic, cfg)
                @views grad[xindices.xindices_dynamic[:xest_to_xdynamic]] .= xdynamic_grad
            else
                _ = _nllh_solveode(xdynamic)
            end
        catch
            fill!(grad, 0.0)
            return nothing
        end
        nxdynamic[1] = tmp
    else
        try
            if nxdynamic[1] != 0
                C = length(cfg.seeds)
                chunk = ForwardDiff.Chunk(C)
                nforward_passes = ceil(nxdynamic[1] / C) |> Int64
                _xdynamic = xdynamic[cache.xdynamic_input_order]
                forwarddiff_gradient_chunks(_nllh_solveode, xdynamic_grad, _xdynamic,
                                            chunk; nforward_passes = nforward_passes)
                @views grad[xindices.xindices_dynamic[:xest_to_xdynamic]] .= xdynamic_grad[cache.xdynamic_output_order]
            else
                _ = _nllh_solveode(xdynamic)
            end
        catch
            fill!(grad, 0.0)
            return nothing
        end
    end

    # In case ODE could not be solved return zero gradient
    if simulation_info.could_solve[1] != true
        fill!(grad, 0.0)
        return nothing
    end

    # None-dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_notode = @view x[xindices.xindices[:not_system_tot]]
    ForwardDiff.gradient!(xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[xindices.xindices[:not_system_tot]] .= xnotode_grad
    return nothing
end

function grad_forward_AD_split!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                                _nllh_solveode::Function, probinfo::PEtabODEProblemInfo,
                                model_info::ModelInfo; cids = [:all],
                                isremade::Bool = false)::Nothing where {T <: AbstractFloat}
    @unpack simulation_info, xindices, priors = model_info
    cache = probinfo.cache
    split_x!(x, xindices, cache; xdynamic_tot = true)
    @unpack xdynamic_grad, xnotode_grad = cache
    _xdynamic_tot = get_tmp(cache.xdynamic_tot, 1.0)

    # Get the Jacobians of Neural-Networks that set values for potential model parameters.
    # By then taking the gradient on the output of these networks, and computing a vjp,
    # the gradient can be computed in minumum number of forward-passes
    _jac_nn_pre_ode!(probinfo)

    # A gradient is computed for each condition-id, only using parameter present for
    # said condition
    fill!(xdynamic_grad, 0.0)
    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        simid = simulation_info.conditionids[:simulation][i]
        ixdynamic_simid = _get_ixdynamic_simid(simid, xindices; nn_pre_ode = false)
        xinput = _get_xinput(simid, _xdynamic_tot, ixdynamic_simid, model_info, probinfo)
        _nllh = (_xinput) -> begin
        _split_xinput!(probinfo, simid, model_info, _xinput, ixdynamic_simid)
            xdynamic_tot = get_tmp(probinfo.cache.xdynamic_tot, _xinput)
            return _nllh_solveode(xdynamic_tot, [cid])
        end
        try
            # The ODE must be solved to get the gradient of the other parameters
            if !isempty(xdynamic_grad)
                _grad = ForwardDiff.gradient(_nllh, xinput)
                xdynamic_grad[ixdynamic_simid] .+= _grad[1:length(ixdynamic_simid)]
                if length(_grad) > length(ixdynamic_simid)
                    cache.grad_nn_pre_ode_outputs .= _grad[(length(ixdynamic_simid)+1):end]
                    _grad_nn_pre_ode!(xdynamic_grad, simid, probinfo, model_info)
                end
            else
                _nllh(xinput)
            end
        catch
            fill!(grad, 0.0)
        end
    end
    if simulation_info.could_solve[1] != true
        fill!(grad, 0.0)
        return nothing
    end
    @views grad[xindices.xindices_dynamic[:xest_to_xdynamic]] .= cache.xdynamic_grad

    x_notode = @view x[xindices.xindices[:not_system_tot]]
    ForwardDiff.gradient!(xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[xindices.xindices[:not_system_tot]] .= xnotode_grad

    # Reset such that neural-nets pre ODE no longer have status of having been evaluated
    _reset_nn_pre_ode!(probinfo)
    return nothing
end

# Compute the gradient via forward sensitivity equations
function grad_forward_eqs!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                           _solve_conditions!::Function, probinfo::PEtabODEProblemInfo,
                           model_info::ModelInfo,
                           cfg::Union{ForwardDiff.JacobianConfig, Nothing};
                           cids::Vector{Symbol} = [:all],
                           isremade::Bool = false)::Nothing where {T <: AbstractFloat}
    @unpack sensealg, cache, split_over_conditions = probinfo
    @unpack priors, xindices = model_info
    @unpack petab_parameters, priors, petab_measurements = model_info
    split_x!(x, xindices, cache; xdynamic_tot = true)

    # See comment above on Jacobian for neural-nets pre ODE-solving
    if probinfo.split_over_conditions == true  || probinfo.sensealg != :ForwardDiff
        _jac_nn_pre_ode!(probinfo)
    end

    _grad_forward_eqs!(cache.xdynamic_grad, _solve_conditions!, probinfo, model_info,
                       cfg; cids = cids, isremade = isremade)
    @views grad[xindices.xindices_dynamic[:xest_to_xdynamic]] .= cache.xdynamic_grad

    # Happens when at least one forward pass fails
    if !isempty(cache.xdynamic_grad) && all(cache.xdynamic_grad .== 0.0)
        fill!(grad, 0.0)
        return nothing
    end

    # None-dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_notode = @view x[xindices.xindices[:not_system_tot]]
    ForwardDiff.gradient!(cache.xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[xindices.xindices[:not_system_tot]] .= cache.xnotode_grad

    # Reset such that neural-nets pre ODE no longer have status of having been evaluated
    _reset_nn_pre_ode!(probinfo)
    return nothing
end
