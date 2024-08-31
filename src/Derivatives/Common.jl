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

function adjust_gradient_θ_transformed!(gradient::Union{AbstractVector, SubArray},
                                        _gradient::AbstractVector,
                                        ∂G∂p::AbstractVector,
                                        xdynamic::Vector{Float64},
                                        θ_indices::ParameterIndices,
                                        simulation_condition_id::Symbol;
                                        autodiff_sensitivites::Bool = false,
                                        adjoint::Bool = false)::Nothing
    map_condition_id = θ_indices.maps_conidition_id[simulation_condition_id]
    map_ode_problem = θ_indices.map_ode_problem

    # Transform gradient parameter that for each experimental condition appear in the ODE system
    i_change = θ_indices.map_ode_problem.sys_to_dynamic
    if autodiff_sensitivites == true
        gradient1 = _gradient[map_ode_problem.sys_to_dynamic] .+
                    ∂G∂p[map_ode_problem.dynamic_to_sys]
    else
        gradient1 = _gradient[map_ode_problem.dynamic_to_sys] .+
                    ∂G∂p[map_ode_problem.dynamic_to_sys]
    end
    @views gradient[i_change] .+= _adjust_gradient_θ_transformed(gradient1,
                                                                 xdynamic[map_ode_problem.sys_to_dynamic],
                                                                 θ_indices.xids[:dynamic][map_ode_problem.sys_to_dynamic],
                                                                 θ_indices)

    # For forward sensitives via autodiff ∂G∂p is on the same scale as ode_problem.p, while
    # S-matrix is on the same scale as xdynamic. To be able to handle condition specific
    # parameters mapping to several ode_problem.p parameters the sensitivity matrix part and
    # ∂G∂p must be treated seperately.
    if autodiff_sensitivites == true
        _ixdynamic = unique(map_condition_id.ix_dynamic)
        gradient[_ixdynamic] .+= _adjust_gradient_θ_transformed(_gradient[_ixdynamic],
                                                                 xdynamic[_ixdynamic],
                                                                 θ_indices.xids[:dynamic][_ixdynamic],
                                                                 θ_indices)
        out = _adjust_gradient_θ_transformed(∂G∂p[map_condition_id.ix_sys],
                                             xdynamic[map_condition_id.ix_dynamic],
                                             θ_indices.xids[:dynamic][map_condition_id.ix_dynamic],
                                             θ_indices)
        @inbounds for i in eachindex(map_condition_id.ix_sys)
            gradient[map_condition_id.ix_dynamic[i]] += out[i]
        end
    end

    # Here both ∂G∂p and _gradient are on the same scale a ode_problem.p. One condition specific parameter
    # can map to several parameters in ode_problem.p
    if adjoint == true || autodiff_sensitivites == false
        out = _adjust_gradient_θ_transformed(_gradient[map_condition_id.ix_sys] .+
                                             ∂G∂p[map_condition_id.ix_sys],
                                             xdynamic[map_condition_id.ix_dynamic],
                                             θ_indices.xids[:dynamic][map_condition_id.ix_dynamic],
                                             θ_indices)
        @inbounds for i in eachindex(map_condition_id.ix_sys)
            gradient[map_condition_id.ix_dynamic[i]] += out[i]
        end
    end

    return nothing
end

function _adjust_gradient_θ_transformed(_gradient::AbstractVector{T},
                                        θ::AbstractVector{T},
                                        n_parameters_estimate::AbstractVector{Symbol},
                                        θ_indices::ParameterIndices)::Vector{T} where {
                                                                                       T <:
                                                                                       Real}
    out = similar(_gradient)
    @inbounds for (i, θ_name) in pairs(n_parameters_estimate)
        θ_scale = θ_indices.θ_scale[θ_name]
        if θ_scale === :log10
            out[i] = log(10) * _gradient[i] * θ[i]
        elseif θ_scale === :log
            out[i] = _gradient[i] * θ[i]
        elseif θ_scale === :lin
            out[i] = _gradient[i]
        end
    end

    return out
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
