# Function to compute ∂G∂u and ∂G∂p for an observation assuming a fixed ODE-solution
function ∂G∂_!(∂G∂_::AbstractVector, u::AbstractVector, p::Vector{T}, t::T, i::Integer,
               imeasurements_t_cid::Vector{Vector{Int64}}, model_info::ModelInfo,
               xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic::Vector{T},
               ∂h∂_::Vector{T}, ∂σ∂_::Vector{T}; ∂G∂U::Bool = true,
               residuals::Bool = false)::Nothing where T <: AbstractFloat
    @unpack measurement_info, θ_indices, parameter_info, petab_model = model_info
    @unpack measurement_transforms, observable_id = measurement_info
    fill!(∂G∂_, 0.0)
    for imeasurement in imeasurements_t_cid[i]
        fill!(∂h∂_, 0.0)
        fill!(∂σ∂_, 0.0)

        h_transformed = computehT(u, t, p, xobservable, xnondynamic, petab_model,
                                  imeasurement, measurement_info, θ_indices, parameter_info)
        σ = computeσ(u, t, p, xnoise, xnondynamic, petab_model, imeasurement,
                     measurement_info, θ_indices, parameter_info)

        mapxnoise = θ_indices.mapxnoise[imeasurement]
        mapxobservable = θ_indices.mapxobservable[imeasurement]
        if ∂G∂U == true
            petab_model.compute_∂h∂u!(u, t, p, xobservable, xnondynamic,
                                      observable_id[imeasurement], mapxobservable, ∂h∂_)
            petab_model.compute_∂σ∂u!(u, t, xnoise, p, xnondynamic, parameter_info,
                                      observable_id[imeasurement], mapxnoise, ∂σ∂_)
        else
            petab_model.compute_∂h∂p!(u, t, p, xobservable, xnondynamic,
                                      observable_id[imeasurement], mapxobservable, ∂h∂_)
            petab_model.compute_∂σ∂p!(u, t, xnoise, p, xnondynamic, parameter_info,
                                      observable_id[imeasurement], mapxnoise, ∂σ∂_)
        end

        measurement_transform = measurement_transforms[imeasurement]
        if measurement_transform == :log10
            ∂h∂_ .*= 1 / (log(10) * exp10(h_transformed))
        elseif measurement_transform == :log
            ∂h∂_ .*= 1 / exp(h_transformed)
        end

        # In case of Guass Newton approximation we target the
        # residuals = (h_transformed - y_transformed) / σ
        y_transformed = measurement_info.measurementT[imeasurement]
        if residuals == false
            ∂G∂h = (h_transformed - y_transformed) / σ^2
            ∂G∂σ = 1 / σ - ((h_transformed - y_transformed)^2 / σ^3)
        else
            ∂G∂h = 1.0 / σ
            ∂G∂σ = -(h_transformed - y_transformed) / σ^2
        end
        @views ∂G∂_ .+= (∂G∂h * ∂h∂_ .+ ∂G∂σ * ∂σ∂_)[:]
    end
    return nothing
end

function _get_∂G∂_!(probleminfo::PEtabODEProblemInfo, model_info::ModelInfo, cid::Symbol,
                    xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic::Vector{T};
                    residuals::Bool = false)::Tuple{Function, Function} where T <: AbstractFloat
    cache = probleminfo.cache
    if residuals == false
        it = model_info.simulation_info.imeasurements_t[cid]
        ∂G∂u! = (out, u, p, t, i) -> begin
            ∂G∂_!(out, u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic,
                  cache.∂h∂u, cache.∂σ∂u; ∂G∂U = true, residuals = residuals)
        end
        ∂G∂p! = (out, u, p, t, i) -> begin
            ∂G∂_!(out, u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic,
                  cache.∂h∂p, cache.∂σ∂p; ∂G∂U = false, residuals = residuals)
        end
    else
        ∂G∂u! = (out, u, p, t, i, it) -> begin
            ∂G∂_!(out, u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic,
                  cache.∂h∂u, cache.∂σ∂u; ∂G∂U = true, residuals = residuals)
        end
        ∂G∂p! = (out, u, p, t, i, it) -> begin
            ∂G∂_!(out, u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic,
                  cache.∂h∂p, cache.∂σ∂p; ∂G∂U = false, residuals = residuals)
        end
    end
    return ∂G∂u!, ∂G∂p!
end

# Adjust the gradient from linear scale to current scale for x-vector
function grad_to_xscale!(grad_xscale, grad_linscale::Vector{T}, ∂G∂p::Vector{T},
                          xdynamic::Vector{T}, θ_indices::ParameterIndices, simid::Symbol;
                          sensitivites_AD::Bool = false, adjoint::Bool = false)::Nothing where T <: AbstractFloat
    @unpack dynamic_to_sys, sys_to_dynamic  = θ_indices.map_odeproblem
    @unpack xids, xscale = θ_indices
    @unpack ix_sys, ix_dynamic = θ_indices.maps_conidition_id[simid]
    # Adjust for parameters that appear in each simulation condition (not unique to simid).
    # Note that ∂G∂p is on the scale of ODEProblem.p which might not be the same scale
    # as parameters appear in the gradient on linear-scale
    if sensitivites_AD == true
        grad_p1 = grad_linscale[sys_to_dynamic] .+ ∂G∂p[dynamic_to_sys]
    else
        grad_p1 = grad_linscale[dynamic_to_sys] .+ ∂G∂p[dynamic_to_sys]
    end
    @views _grad_to_xscale!(grad_xscale[sys_to_dynamic], grad_p1, xdynamic[sys_to_dynamic],
                            xids[:dynamic][sys_to_dynamic], xscale)

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
        for (i, imap) in pairs(ix_sys)
            grad_xscale[imap] += out[i]
        end
    end

    # Here both ∂G∂p and grad_linscale are on the same scale a ODEProblem.p. Again
    # condition specific variables can map to several parameters in the ODESystem,
    # thus the for-loop
    if adjoint == true || sensitivites_AD == false
        @views _grad_to_xscale!(grad_xscale[ix_sys], grad_linscale[ix_sys],
                                xdynamic[ix_dynamic], xids[:dynamic][ix_dynamic], xscale)
        @views out = _grad_to_xscale(grad_linscale[ix_sys] .+ ∂G∂p[ix_sys],  xdynamic[ix_dynamic],
                                     xids[:dynamic][ix_dynamic], xscale)
        for (i, imap) in pairs(ix_sys)
            grad_xscale[imap] += out[i]
        end
    end
    return nothing
end

function _grad_to_xscale!(grad_xscale::AbstractVector{T}, grad_linscale::AbstractVector{T},
                          x::AbstractVector{T}, xids::AbstractVector{Symbol},
                          xscale::Dict{Symbol, Symbol})::Nothing where T <: AbstractFloat
    for (i, xid) in pairs(xids)
        grad_xscale[i] += _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return nothing
end

function _grad_to_xscale(grad_linscale::AbstractVector{T}, x::AbstractVector{T},
                         xids::AbstractVector{Symbol}, xscale::Dict{Symbol, Symbol})::Vector{T} where T <: AbstractFloat
    grad_xscale = similar(grad_linscale)
    for (i, xid) in pairs(xids)
        grad_xscale[i] = _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return grad_xscale
end
function _grad_to_xscale(grad_linscale_val::T, x_val::T, xscale::Symbol)::T where T <: AbstractFloat
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
