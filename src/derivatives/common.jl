function _G(u::AbstractVector, p::AbstractVector, t::T, i::Integer,
            imeasurements_t_cid::Vector{Vector{Int64}}, model_info::ModelInfo,
            xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic::Vector{T},
            residuals::Bool) where {T <: AbstractFloat}
    @unpack petab_measurements, xindices, petab_parameters, model = model_info
    @unpack measurement_transforms, observable_id = petab_measurements
    nominal_values = petab_parameters.nominal_value
    out = 0.0
    for imeasurement in imeasurements_t_cid[i]
        obsid = observable_id[imeasurement]
        xnoise_maps = xindices.xnoise_maps[imeasurement]
        xobservable_maps = xindices.xobservable_maps[imeasurement]
        h = _h(u, t, p, xobservable, xnondynamic, model.h, xobservable_maps, obsid,
               nominal_values)
        h_transformed = transform_observable(h, measurement_transforms[imeasurement])
        σ = _sd(u, t, p, xnoise, xnondynamic, model.sd, xnoise_maps, obsid, nominal_values)

        y_transformed = petab_measurements.measurement_transformed[imeasurement]
        residual = (h_transformed - y_transformed) / σ
        if residuals == true
            out += residual
            continue
        end
        out += _nllh_obs(residual, σ, y_transformed, measurement_transforms[imeasurement])
    end
    return out
end

function _get_∂G∂_!(probinfo::PEtabODEProblemInfo, model_info::ModelInfo, cid::Symbol,
                    xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic::Vector{T};
                    residuals::Bool = false)::Tuple{Function,
                                                    Function} where {T <: AbstractFloat}
    if residuals == false
        it = model_info.simulation_info.imeasurements_t[cid]
        ∂G∂u! = (out, u, p, t, i) -> begin
            _fu = (u) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic, false)
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i) -> begin
            _fp = (p) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic, false)
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    else
        ∂G∂u! = (out, u, p, t, i, it) -> begin
            _fu = (u) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic, true)
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i, it) -> begin
            _fp = (p) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic, true)
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

function _get_ixdynamic_simid(simid::Symbol, xindices::ParameterIndices; full_x::Bool = false)::Vector{Integer}
    @unpack isys_all_conditions, ix_condition = xindices.condition_maps[simid]
    if full_x == false
        ixdynamic = Iterators.flatten((isys_all_conditions, ix_condition))
    else
        ixdynamic = Iterators.flatten((isys_all_conditions, ix_condition, xindices.xindices[:not_system]))
    end
    return unique(ixdynamic)
end
