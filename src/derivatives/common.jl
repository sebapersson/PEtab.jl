function _G(u::AbstractVector, p, t::T, i::Integer, imeasurements_t_cid::Vector{Vector{Int64}}, model_info::ModelInfo, xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic_mech::Vector{T}, xnn::Dict{Symbol, ComponentArray}, xnn_constant::Dict{Symbol, ComponentArray}, residuals::Bool) where {T <: AbstractFloat}
    @unpack petab_measurements, xindices, petab_parameters, model = model_info
    @unpack measurements_transformed, measurements, noise_distributions, observable_id = petab_measurements
    @unpack nominal_value = petab_parameters

    out = 0.0
    for im in imeasurements_t_cid[i]
        obsid = observable_id[im]
        xnoise_maps = xindices.xnoise_maps[im]
        xobservable_maps = xindices.xobservable_maps[im]

        h = _h(u, t, p, xobservable, xnondynamic, model, xobservable_maps, obsid,
               nominal_value)
        σ = _sd(u, t, p, xnoise, xnondynamic, model, xnoise_maps, obsid, nominal_value)

        if residuals == true
            h_transformed = _transform_h(h, noise_distributions[im])
            residual = (h_transformed - measurements_transformed[im]) / σ
            out += residual
            continue
        end
        out += _nllh_obs(h, σ, measurements[im], noise_distributions[im])
    end
    return out
end

function _get_∂G∂_!(model_info::ModelInfo, cid::Symbol, xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic_mech::Vector{T}, xnn::Dict{Symbol, ComponentArray}, xnn_constant::Dict{Symbol, ComponentArray}; residuals::Bool = false)::NTuple{2, Function} where {T <: AbstractFloat}
    if residuals == false
        it = model_info.simulation_info.imeasurements_t[cid]
        ∂G∂u! = (out, u, p, t, i) -> begin
            _fu = (u) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, xnn_constant, false)
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i) -> begin
            _fp = (p) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, xnn_constant, false)
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    else
        ∂G∂u! = (out, u, p, t, i, it) -> begin
            _fu = (u) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, xnn_constant, true)
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i, it) -> begin
            _fp = (p) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, xnn_constant, true)
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    end
    return ∂G∂u!, ∂G∂p!
end

#
function grad_to_xscale!(grad_xscale, grad_linscale::Vector{T}, ∂G∂p::Vector{T}, xdynamic::Vector{T}, xindices::ParameterIndices, simid::Symbol; sensitivities_AD::Bool = false)::Nothing where {T <: AbstractFloat}
    @unpack xids, xscale, condition_maps = xindices
    condition_map! = condition_maps[simid]

    # TODO: Jacobian should be pre-allocated, doable as I should have all the dimensions when building the ConditionMap
    # In case of ForwardDiff sensitivities (sensitivities_AD == true) grad_linscale follow
    # the same indexing as grad_xscale (that of xdynamic) with xdynamic mapped to p within
    # an AD call. ∂G∂p follows the same indexing as ODEProblem.p, where xdynamic is
    # mapped to p for ∂G∂p outside an AD call -> Jacobian correction needed
    if sensitivities_AD == true
        J = ForwardDiff.jacobian(condition_map!, similar(∂G∂p), xdynamic)
        grad_linscale .+= transpose(transpose(∂G∂p) * J)
        _grad_to_xscale!(grad_xscale, grad_linscale, xdynamic, xids[:dynamic], xscale)
        return nothing
    end

    # In case of sensitivities_AD == false, but grad_linscale and ∂G∂p follow the same
    # indexing as ODEProblem.p, with xdynamic mapped to p outside AD calls -> Jacobian
    # correction needed for both
    J = ForwardDiff.jacobian(condition_map!, similar(∂G∂p), xdynamic)
    _grad_linscale = transpose(transpose(grad_linscale + ∂G∂p) * J)
    _grad_to_xscale!(grad_xscale, _grad_linscale, xdynamic, xids[:dynamic], xscale)
    return nothing
end

function _grad_to_xscale!(grad_xscale::AbstractVector{T}, grad_linscale::AbstractVector{T},
                          x::AbstractVector{T}, xids::AbstractVector{Symbol},
                          xscale::Dict{Symbol, Symbol})::Nothing where {T <: AbstractFloat}
    for (i, xid) in pairs(xids)
        grad_xscale[i] += _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return nothing
end

function _grad_to_xscale(grad_linscale::T, x::T, xscale::Symbol)::T where {T <: AbstractFloat}
    if xscale === :log10
        return log(10) * grad_linscale * x
    elseif xscale === :lin
        return grad_linscale
    elseif xscale == :log2
        return log(2) * grad_linscale * x
    elseif xscale === :log
        return grad_linscale * x
    end
end

function _could_solveode_nllh(simulation_info::SimulationInfo)::Bool
    for cid in simulation_info.conditionids[:experiment]
        sol = simulation_info.odesols[cid]
        if !(sol.retcode in [ReturnCode.Success, ReturnCode.Terminated])
            return false
        end
    end
    return true
end

function _get_xinput(simid::Symbol, x::Vector{<:AbstractFloat}, ixdynamic_simid, model_info::ModelInfo, probinfo::PEtabODEProblemInfo)
    @unpack xindices, simulation_info = model_info
    ninode, npreode = length(ixdynamic_simid), length(xindices.xids[:ml_preode_outputs])
    xinput = zeros(Float64, ninode + npreode)
    @views xinput[1:ninode] .= x[ixdynamic_simid]
    if isempty(probinfo.ml_models_pre_ode)
        return xinput
    end
    for (ml_model_id, nn_preode) in probinfo.ml_models_pre_ode[simid]
        map_ml_model = model_info.xindices.maps_nn_preode[simid][ml_model_id]
        outputs = get_tmp(nn_preode.outputs, xinput)
        ix = map_ml_model.ix_nn_outputs_grad
        ix .= map_ml_model.ix_nn_outputs .+ ninode
        @views xinput[ix] .= outputs
    end
    return xinput
end

function _split_xinput!(probinfo::PEtabODEProblemInfo, simid::Symbol, model_info::ModelInfo, xinput::AbstractVector, ixdynamic_simid::Vector{Integer})::Nothing
    xdynamic_tot = get_tmp(probinfo.cache.xdynamic_tot, xinput)
    @views xdynamic_tot[ixdynamic_simid] .= xinput[1:length(ixdynamic_simid)]
    if isempty(probinfo.ml_models_pre_ode)
        return nothing
    end
    for (ml_model_id, nn_preode) in probinfo.ml_models_pre_ode[simid]
        map_ml_model =  model_info.xindices.maps_nn_preode[simid][ml_model_id]
        outputs = get_tmp(nn_preode.outputs, xinput)
        @views outputs .= xinput[map_ml_model.ix_nn_outputs_grad]
    end
    return nothing
end
