function grad_forward_AD!(
        grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function, _nllh_solveode::Function,
        cfg::ForwardDiff.GradientConfig, probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
        cids::Vector{Symbol} = [:all]
    )::Nothing where {T <: AbstractFloat}
    @unpack simulation_info, xindices, priors = model_info
    @unpack cache = probinfo
    @unpack xdynamic_grad, x_not_system_grad = cache

    # As a subset of ForwardDiff chunks might fail, return code status is checked via
    # simulation info
    simulation_info.could_solve[1] = true

    fill!(grad, 0.0)
    split_x!(x, xindices, cache; xdynamic_full = true)
    xdynamic = get_tmp(cache.xdynamic, x)
    #try
        # In case of no length(xdynamic) = 0 the ODE must still be solved to get
        # the gradient of nondynamic parameters
        if length(xdynamic_grad) != 0
            ForwardDiff.gradient!(xdynamic_grad, _nllh_solveode, xdynamic, cfg)
            @views grad[xindices.indices_est[:est_to_dynamic]] .= xdynamic_grad
        else
            _ = _nllh_solveode(xdynamic)
        end
    #catch
    #    fill!(grad, 0.0)
    #    return nothing
    #end

    # In case ODE could not be solved return zero gradient
    if simulation_info.could_solve[1] != true
        fill!(grad, 0.0)
        return nothing
    end

    # None-dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_not_system = @view x[xindices.indices_est[:est_to_not_system]]
    ForwardDiff.gradient!(x_not_system_grad, _nllh_not_solveode, x_not_system)
    @views grad[xindices.indices_est[:est_to_not_system]] .= x_not_system_grad
    return nothing
end

function grad_forward_AD_split!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                                _nllh_solveode::Function, probinfo::PEtabODEProblemInfo,
                                model_info::ModelInfo; cids = [:all])::Nothing where {T <: AbstractFloat}
    @unpack simulation_info, xindices, priors = model_info
    cache = probinfo.cache
    split_x!(x, xindices, cache; xdynamic_full = true)
    @unpack xdynamic_grad, x_not_system_grad = cache
    _xdynamic = get_tmp(cache.xdynamic, 1.0)

    # Get the Jacobian of Neural-Networks that set values for potential model parameters.
    # By then taking the gradient on the output of these networks, and computing a vjp,
    # the gradient can be computed in minumum number of forward-passes
    _jac_ml_model_pre_simulate!(probinfo, model_info)

    # A gradient is computed for each condition-id, only using parameter present for
    # said condition
    simulation_info.could_solve[1] = true
    fill!(xdynamic_grad, 0.0)
    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        simid = simulation_info.conditionids[:simulation][i]
        ixdynamic_simid = _get_ixdynamic_simid(simid, xindices; ml_pre_simulate = false)
        xinput = _get_xinput(simid, _xdynamic, ixdynamic_simid, model_info, probinfo)
        _nllh = (_xinput) -> begin
            _split_xinput!(probinfo, simid, model_info, _xinput, ixdynamic_simid)
            xdynamic = get_tmp(probinfo.cache.xdynamic, _xinput)
            return _nllh_solveode(xdynamic, [cid])
        end
        try
            # The ODE must be solved to get the gradient of the other parameters
            if !isempty(xdynamic_grad)
                _grad = ForwardDiff.gradient(_nllh, xinput)
                xdynamic_grad[ixdynamic_simid] .+= _grad[1:length(ixdynamic_simid)]
                if length(_grad) > length(ixdynamic_simid)
                    cache.grad_ml_pre_simulate_outputs .= _grad[(length(ixdynamic_simid)+1):end]
                    _set_grax_x_ml_pre_simulate!(xdynamic_grad, simid, probinfo, model_info)
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
    @views grad[xindices.indices_est[:est_to_dynamic]] .= cache.xdynamic_grad

    x_not_system = @view x[xindices.indices_est[:est_to_not_system]]
    ForwardDiff.gradient!(x_not_system_grad, _nllh_not_solveode, x_not_system)
    @views grad[xindices.indices_est[:est_to_not_system]] .= x_not_system_grad

    # Reset such that neural-nets pre ODE no longer have status of having been evaluated
    reset_ml_pre_simulate!(probinfo)
    return nothing
end

# Compute the gradient via forward sensitivity equations
function grad_forward_eqs!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                           _solve_conditions!::Function, probinfo::PEtabODEProblemInfo,
                           model_info::ModelInfo,
                           cfg::Union{ForwardDiff.JacobianConfig, Nothing};
                           cids::Vector{Symbol} = [:all])::Nothing where {T <: AbstractFloat}
    @unpack sensealg, cache, split_over_conditions = probinfo
    @unpack priors, xindices = model_info
    @unpack petab_parameters, priors, petab_measurements = model_info
    split_x!(x, xindices, cache; xdynamic_full = true)

    # See comment above on Jacobian for neural-nets pre ODE-solving
    if probinfo.split_over_conditions == true  || probinfo.sensealg != :ForwardDiff
        _jac_ml_model_pre_simulate!(probinfo, model_info)
    end

    _grad_forward_eqs!(cache.xdynamic_grad, _solve_conditions!, probinfo, model_info,
                       cfg; cids = cids)
    @views grad[xindices.indices_est[:est_to_dynamic]] .= cache.xdynamic_grad

    # Happens when at least one forward pass fails
    if !isempty(cache.xdynamic_grad) && all(cache.xdynamic_grad .== 0.0)
        fill!(grad, 0.0)
        return nothing
    end

    # None-dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_not_system = @view x[xindices.indices_est[:est_to_not_system]]
    ForwardDiff.gradient!(cache.x_not_system_grad, _nllh_not_solveode, x_not_system)
    @views grad[xindices.indices_est[:est_to_not_system]] .= cache.x_not_system_grad

    # Reset such that neural-nets pre ODE no longer have status of having been evaluated
    reset_ml_pre_simulate!(probinfo)
    return nothing
end
