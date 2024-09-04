function grad_forward_AD!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                          _nllh_solveode::Function, cfg::ForwardDiff.GradientConfig,
                          probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                          cids::Vector{Symbol} = [:all], isremade::Bool = false)::Nothing where T <: AbstractFloat
    cache = probleminfo.cache
    @unpack simulation_info, θ_indices, prior_info = model_info
    @unpack xdynamic_grad, xnotode_grad, xdynamic, nxdynamic = cache

    # As a subset of ForwardDiff chunks might fail, return code status is checked via
    # simulation info
    simulation_info.could_solve[1] = true

    # When remaking the problem order of parameters a subset of xdynamic is fixed which
    # must be accounted for in gradient computations
    fill!(grad, 0.0)
    split_x!(x, θ_indices, cache)
    if isremade == false || length(xdynamic) == nxdynamic[1]
        tmp = nxdynamic[1]
        nxdynamic[1] = length(xdynamic)
        try
            # In case of no length(xdynamic) = 0 the ODE must still be solved to get
            # the gradient of nondynamic parameters
            if length(xdynamic_grad) != 0
                ForwardDiff.gradient!(xdynamic_grad, _nllh_solveode, xdynamic, cfg)
                @views grad[θ_indices.xindices[:dynamic]] .= xdynamic_grad
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
                @views grad[θ_indices.xindices[:dynamic]] .= xdynamic_grad[cache.xdynamic_output_order]
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
    x_notode = @view x[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[θ_indices.xindices[:not_system]] .= xnotode_grad
    return nothing
end

function grad_forward_AD_split!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                                _nllh_solveode::Function, probleminfo::PEtabODEProblemInfo,
                                model_info::ModelInfo; cids = [:all],
                                isremade::Bool = false)::Nothing where T <: AbstractFloat
    @unpack simulation_info, θ_indices, prior_info = model_info
    cache = probleminfo.cache
    split_x!(x, θ_indices, cache)
    @unpack xdynamic, xdynamic_grad, xnotode_grad = cache

    # A gradient is computed for each condition-id, only using parameter present for
    # said condition
    fill!(xdynamic_grad, 0.0)
    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        simid = simulation_info.conditionids[:simulation][i]
        ixdynamic_simid = _get_ixdynamic_simid(simid, θ_indices)
        xinput = x[ixdynamic_simid]

        _nllh = (_xinput) -> begin
            _xdynamic = convert.(eltype(_xinput), xdynamic)
            @views _xdynamic[ixdynamic_simid] .= _xinput
            return _nllh_solveode(_xdynamic, [cid])
        end
        try
            # The ODE must be solved to get the gradient of the other parameters
            if !isempty(xdynamic)
                _grad = ForwardDiff.gradient(_nllh, xinput)
                xdynamic_grad[ixdynamic_simid] .+= _grad
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
    @views grad[θ_indices.xindices[:dynamic]] .= xdynamic_grad

    x_notode = @view x[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[θ_indices.xindices[:not_system]] .= xnotode_grad
    return nothing
end

# Compute the gradient via forward sensitivity equations
function grad_forward_eqs!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                           _solve_conditions!::Function, probleminfo::PEtabODEProblemInfo,
                           model_info::ModelInfo, cfg::Union{ForwardDiff.JacobianConfig, Nothing};
                           cids::Vector{Symbol} = [:all], isremade::Bool = false)::Nothing where T <: AbstractFloat
    @unpack sensealg, cache, split_over_conditions = probleminfo
    @unpack prior_info, θ_indices = model_info
    @unpack parameter_info, prior_info, measurement_info = model_info
    split_x!(x, θ_indices, cache)

    _grad_forward_eqs!(cache.xdynamic_grad, _solve_conditions!, probleminfo, model_info,
                       cfg; cids = cids, isremade = isremade)
    @views grad[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad

    # Happens when at least one forward pass fails
    if !isempty(cache.xdynamic_grad) && all(cache.xdynamic_grad .== 0.0)
        fill!(grad, 0.0)
        return nothing
    end

    # None-dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_notode = @view x[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(cache.xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[θ_indices.xindices[:not_system]] .= cache.xnotode_grad
    return nothing
end
