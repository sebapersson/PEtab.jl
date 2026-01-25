function _G(
        u::AbstractVector, p, t::T, i::Integer, imeasurements_t_cid::Vector{Vector{Int64}},
        model_info::ModelInfo, xnoise::Vector{T}, xobservable::Vector{T},
        xnondynamic_mech::Vector{T}, x_ml_models::Dict{Symbol, ComponentArray},
        x_ml_models_constant::Dict{Symbol, ComponentArray}, residuals::Bool
    ) where {T <: AbstractFloat}
    @unpack petab_measurements, xindices, petab_parameters, model = model_info
    @unpack measurements_transformed, measurements, noise_distributions, observable_id = petab_measurements
    @unpack nominal_value = petab_parameters

    out = 0.0
    for im in imeasurements_t_cid[i]
        obsid = observable_id[im]
        xnoise_maps = xindices.xnoise_maps[im]
        xobservable_maps = xindices.xobservable_maps[im]

        h = _h(
            u, t, p, xobservable, xnondynamic_mech, x_ml_models, x_ml_models_constant,
            model, xobservable_maps, obsid, nominal_value
        )
        σ = _sd(
            u, t, p, xnoise, xnondynamic_mech, x_ml_models, x_ml_models_constant, model,
            xnoise_maps, obsid, nominal_value
        )

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

function _get_∂G∂_!(
        model_info::ModelInfo, cid::Symbol, xnoise::Vector{T}, xobservable::Vector{T},
        xnondynamic_mech::Vector{T}, x_ml_models::Dict{Symbol, ComponentArray},
        x_ml_models_constant::Dict{Symbol, ComponentArray}; residuals::Bool = false
    )::NTuple{2, Function} where {T <: AbstractFloat}
    if residuals == false
        it = model_info.simulation_info.imeasurements_t[cid]
        ∂G∂u! = (out, u, p, t, i) -> begin
            _fu = (u) -> begin
                _G(
                    u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech,
                    x_ml_models, x_ml_models_constant, false
                )
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i) -> begin
            _fp = (p) -> begin
                _G(
                    u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech,
                    x_ml_models, x_ml_models_constant, false
                )
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    else
        ∂G∂u! = (out, u, p, t, i, it) -> begin
            _fu = (u) -> begin
                _G(
                    u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech,
                    x_ml_models, x_ml_models_constant, true
                )
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i, it) -> begin
            _fp = (p) -> begin
                _G(
                    u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech,
                    x_ml_models, x_ml_models_constant, true
                )
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    end
    return ∂G∂u!, ∂G∂p!
end

#
function grad_to_xscale!(
        grad_xscale, grad_linscale::Vector{T}, ∂G∂p::Vector{T}, xdynamic::Vector{T},
        xindices::ParameterIndices, simid::Symbol; sensitivities_AD::Bool = false,
        ml_pre_simulate::Bool = false
    )::Nothing where {T <: AbstractFloat}
    @unpack ids, xscale, condition_maps = xindices
    condition_map! = condition_maps[simid]

    # Neural net parameters for a neural net before the ODE dynamics. These gradient
    # components only considered for ForwardEquations full AD, where it is not possible to
    # use the chain-rule to separately differentitate each componenent, and these parameters
    # are differentiated using full AD.
    if ml_pre_simulate == true
        xi = xindices.indices_dynamic[:dynamic_to_ml_pre_simulate]
        @views grad_xscale[xi] .+= grad_linscale[xi]
    end

    # ML parameters inside the ODE. These should not be transformed to parameters scale
    # (only allowed to be on linear scale)
    @unpack dynamic_to_ml_sys, sys_to_dynamic_ml = xindices.indices_dynamic
    if sensitivities_AD == true
        @views @. grad_xscale[dynamic_to_ml_sys] += (
            grad_linscale[dynamic_to_ml_sys] + ∂G∂p[sys_to_dynamic_ml]
        )
    else
        @views @. grad_xscale[dynamic_to_ml_sys] += (
            grad_linscale[sys_to_dynamic_ml] + ∂G∂p[sys_to_dynamic_ml]
        )
    end

    # In case of ForwardDiff sensitivities (sensitivities_AD == true) grad_linscale follow
    # the same indexing as grad_xscale (that of xdynamic) with xdynamic mapped to p within
    # an AD call. ∂G∂p follows the same indexing as ODEProblem.p, where xdynamic is
    # mapped to p for ∂G∂p outside an AD call -> Jacobian correction needed
    if sensitivities_AD == true && ml_pre_simulate == true
        J = ForwardDiff.jacobian(condition_map!, similar(∂G∂p), xdynamic)
        grad_linscale .+= transpose(transpose(∂G∂p) * J)
        _grad_to_xscale!(
            grad_xscale, grad_linscale, xdynamic, ids[:est_to_dynamic_mech], xscale
        )
        return nothing

    elseif sensitivities_AD == true && ml_pre_simulate == false
        n_dynamic_mech = length(ids[:est_to_dynamic_mech])
        _xdynamic = xdynamic[1:n_dynamic_mech]
        J = ForwardDiff.jacobian(condition_map!, similar(∂G∂p), _xdynamic)
        grad_linscale[1:n_dynamic_mech] .+= transpose(transpose(∂G∂p) * J)
        _grad_to_xscale!(
            grad_xscale, grad_linscale, xdynamic, ids[:est_to_dynamic_mech], xscale
        )
        return nothing
    end

    # In case of sensitivities_AD == false, grad_linscale and ∂G∂p follow the same
    # indexing as ODEProblem.p, with xdynamic mapped to p outside AD calls -> Jacobian
    # correction needed for both
    J = ForwardDiff.jacobian(condition_map!, similar(∂G∂p), xdynamic)
    _grad_linscale = transpose(transpose(grad_linscale + ∂G∂p) * J)
    _grad_to_xscale!(
        grad_xscale, _grad_linscale, xdynamic, ids[:est_to_dynamic_mech], xscale
    )
    return nothing
end

function _grad_to_xscale!(
        grad_xscale::AbstractVector{T}, grad_linscale::AbstractVector{T},
        x::AbstractVector{T}, ids::AbstractVector{Symbol}, xscale::Dict{Symbol, Symbol}
    )::Nothing where {T <: AbstractFloat}
    for (i, xid) in pairs(ids)
        grad_xscale[i] += _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return nothing
end

function _grad_to_xscale(
        grad_linscale::T, x::T, xscale::Symbol
    )::T where {T <: AbstractFloat}
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

function _get_xinput(
        simid::Symbol, x::Vector{<:AbstractFloat}, ixdynamic_simid, model_info::ModelInfo,
        probinfo::PEtabODEProblemInfo
    )
    @unpack xindices, simulation_info = model_info
    ninode = length(ixdynamic_simid)
    npre_simulate = length(xindices.ids[:sys_ml_pre_simulate_outputs])
    xinput = zeros(Float64, ninode + npre_simulate)
    @views xinput[1:ninode] .= x[ixdynamic_simid]

    if isempty(probinfo.ml_models_pre_ode)
        return xinput
    end

    for (ml_id, ml_model_pre_simulate) in probinfo.ml_models_pre_ode[simid]
        map_ml_model = model_info.xindices.maps_ml_pre_simulate[simid][ml_id]
        outputs = get_tmp(ml_model_pre_simulate.outputs, xinput)
        ix = map_ml_model.ix_ml_outputs .+ ninode
        @views xinput[ix] .= outputs
    end
    return xinput
end

function _split_xinput!(
        probinfo::PEtabODEProblemInfo, simid::Symbol, model_info::ModelInfo,
        xinput::AbstractVector, ixdynamic_simid::Vector{Integer}
    )::Nothing
    xdynamic = get_tmp(probinfo.cache.xdynamic, xinput)
    @views xdynamic[ixdynamic_simid] .= xinput[1:length(ixdynamic_simid)]

    if isempty(probinfo.ml_models_pre_ode)
        return nothing
    end

    for (ml_id, ml_model_pre_simulate) in probinfo.ml_models_pre_ode[simid]
        map_ml_model = model_info.xindices.maps_ml_pre_simulate[simid][ml_id]
        outputs = get_tmp(ml_model_pre_simulate.outputs, xinput)
        ix = map_ml_model.ix_ml_outputs .+ length(ixdynamic_simid)
        @views outputs .= xinput[ix]
    end
    return nothing
end
