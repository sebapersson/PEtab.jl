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
        mapxnoise = xindices.mapxnoise[imeasurement]
        mapxobservable = xindices.mapxobservable[imeasurement]
        h = _h(u, t, p, xobservable, xnondynamic, model.h, mapxobservable, obsid,
               nominal_values)
        h_transformed = transform_observable(h, measurement_transforms[imeasurement])
        σ = _sd(u, t, p, xnoise, xnondynamic, model.sd, mapxnoise, obsid, nominal_values)

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

# Adjust the gradient from linear scale to current scale for x-vector
function grad_to_xscale!(grad_xscale, grad_linscale::Vector{T}, ∂G∂p::Vector{T},
                         xdynamic::Vector{T}, xindices::ParameterIndices, simid::Symbol;
                         sensitivites_AD::Bool = false,
                         adjoint::Bool = false)::Nothing where {T <: AbstractFloat}
    @unpack dynamic_to_sys, sys_to_dynamic = xindices.map_odeproblem
    @unpack xids, xscale = xindices
    @unpack ix_sys, ix_dynamic = xindices.maps_conidition_id[simid]
    # Adjust for parameters that appear in each simulation condition (not unique to simid).
    # Note that ∂G∂p is on the scale of ODEProblem.p which might not be the same scale
    # as parameters appear in the gradient on linear-scale
    if sensitivites_AD == true
        grad_p1 = grad_linscale[dynamic_to_sys] .+ ∂G∂p[sys_to_dynamic]
    else
        grad_p1 = grad_linscale[sys_to_dynamic] .+ ∂G∂p[sys_to_dynamic]
    end
    @views _grad_to_xscale!(grad_xscale[dynamic_to_sys], grad_p1, xdynamic[dynamic_to_sys],
                            xids[:dynamic][dynamic_to_sys], xscale)

    # For forward sensitives via autodiff ∂G∂p is on the same scale as ode_problem.p, while
    # S-matrix is on the same scale as xdynamic. To be able to handle condition specific
    # parameters mapping to several ode_problem.p parameters the sensitivity matrix part
    # and ∂G∂p must be treated seperately. Further, as condition specific variables can map
    # to several parameters in the ODESystem, the for-loop is needed
    if sensitivites_AD == true
        ix = unique(ix_dynamic)
        @views _grad_to_xscale!(grad_xscale[ix], grad_linscale[ix], xdynamic[ix],
                                xids[:dynamic][ix], xscale)
        @views out = _grad_to_xscale(∂G∂p[ix_sys], xdynamic[ix_dynamic],
                                     xids[:dynamic][ix_dynamic], xscale)
        for (i, imap) in pairs(ix)
            grad_xscale[imap] += out[i]
        end
    end

    # Here both ∂G∂p and grad_linscale are on the same scale a ODEProblem.p. Again
    # condition specific variables can map to several parameters in the ODESystem,
    # thus the for-loop
    if adjoint == true || sensitivites_AD == false
        @views _grad_to_xscale!(grad_xscale[ix_sys], grad_linscale[ix_sys],
                                xdynamic[ix_dynamic], xids[:dynamic][ix_dynamic], xscale)
        @views out = _grad_to_xscale(grad_linscale[ix_sys] .+ ∂G∂p[ix_sys],
                                     xdynamic[ix_dynamic],
                                     xids[:dynamic][ix_dynamic], xscale)
        for (i, imap) in pairs(ix_sys)
            grad_xscale[imap] += out[i]
        end
    end
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

function _grad_to_xscale(grad_linscale::AbstractVector{T}, x::AbstractVector{T},
                         xids::AbstractVector{Symbol},
                         xscale::Dict{Symbol, Symbol})::Vector{T} where {T <: AbstractFloat}
    grad_xscale = similar(grad_linscale)
    for (i, xid) in pairs(xids)
        grad_xscale[i] = _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return grad_xscale
end
function _grad_to_xscale(grad_linscale_val::T, x_val::T,
                         xscale::Symbol)::T where {T <: AbstractFloat}
    if xscale === :log10
        return log(10) * grad_linscale_val * x_val
    elseif xscale === :log
        return grad_linscale_val * x_val
    elseif xscale === :lin
        return grad_linscale_val
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
