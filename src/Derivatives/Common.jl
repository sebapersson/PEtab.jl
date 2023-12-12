function get_index_parameters_not_ODE(θ_indices::ParameterIndices)::Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}}

    @unpack θ_observable_names, θ_sd_names, θ_non_dynamic_names, θ_not_ode_names = θ_indices

    iθ_sd::Vector{Int64} = [findfirst(x -> x == θ_sd_names[i], θ_not_ode_names) for i in eachindex(θ_sd_names)]
    iθ_observable::Vector{Int64} = [findfirst(x -> x == θ_observable_names[i],  θ_not_ode_names) for i in eachindex(θ_observable_names)]
    iθ_non_dynamic::Vector{Int64} = [findfirst(x -> x ==  θ_non_dynamic_names[i],  θ_not_ode_names) for i in eachindex(θ_non_dynamic_names)]
    iθ_not_ode::Vector{Int64} = θ_indices.iθ_not_ode

    return iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode
end


# Function to compute ∂G∂u and ∂G∂p for an observation assuming a fixed ODE-solution
function compute∂G∂_(∂G∂_,
                     u::AbstractVector,
                     p::Vector{Float64}, # ode_problem.p
                     t::Float64,
                     i::Integer,
                     i_per_time_point::Vector{Vector{Int64}},
                     measurement_info::MeasurementsInfo,
                     parameter_info::ParametersInfo,
                     θ_indices::ParameterIndices,
                     petab_model::PEtabModel,
                     θ_sd::Vector{Float64},
                     θ_observable::Vector{Float64},
                     θ_non_dynamic::Vector{Float64},
                     ∂h∂_::Vector{Float64},
                     ∂σ∂_::Vector{Float64};
                     compute∂G∂U::Bool=true,
                     compute_residuals::Bool=false)::Nothing

    fill!(∂G∂_, 0.0)
    for i_measurement_data in i_per_time_point[i]
        fill!(∂h∂_, 0.0)
        fill!(∂σ∂_, 0.0)

        hT = computehT(u, t, p, θ_observable,  θ_non_dynamic, petab_model, i_measurement_data, measurement_info, θ_indices, parameter_info)
        σ = computeσ(u, t, p, θ_sd,  θ_non_dynamic, petab_model, i_measurement_data, measurement_info, θ_indices, parameter_info)

        # Maps needed to correctly extract the right SD and observable parameters
        mapθ_sd = θ_indices.mapθ_sd[i_measurement_data]
        mapθ_observable = θ_indices.mapθ_observable[i_measurement_data]
        if compute∂G∂U == true
            petab_model.compute_∂h∂u!(u, t, p, θ_observable,  θ_non_dynamic, measurement_info.observable_id[i_measurement_data], mapθ_observable, ∂h∂_)
            petab_model.compute_∂σ∂u!(u, t, θ_sd, p,  θ_non_dynamic, parameter_info, measurement_info.observable_id[i_measurement_data], mapθ_sd, ∂σ∂_)
        else
            petab_model.compute_∂h∂p!(u, t, p, θ_observable,  θ_non_dynamic, measurement_info.observable_id[i_measurement_data], mapθ_observable, ∂h∂_)
            petab_model.compute_∂σ∂p!(u, t, θ_sd, p,  θ_non_dynamic, parameter_info, measurement_info.observable_id[i_measurement_data], mapθ_sd, ∂σ∂_)
        end

        if measurement_info.measurement_transformation[i_measurement_data] === :log10
            y_obs = measurement_info.measurementT[i_measurement_data]
            ∂h∂_ .*= 1 / (log(10) * exp10(hT))
        elseif measurement_info.measurement_transformation[i_measurement_data] === :log
            y_obs = measurement_info.measurementT[i_measurement_data]
            ∂h∂_ .*= 1 / exp(hT)
        elseif measurement_info.measurement_transformation[i_measurement_data] === :lin
            y_obs = measurement_info.measurement[i_measurement_data]
        end

        # In case of Guass Newton approximation we target the residuals (y_mod - y_obs)/σ
        if compute_residuals == false
            ∂G∂h = ( hT - y_obs ) / σ^2
            ∂G∂σ = 1/σ - (( hT - y_obs )^2 / σ^3)
        else
            ∂G∂h = 1.0 / σ
            ∂G∂σ = -(hT - y_obs) / σ^2
        end

        @views ∂G∂_ .+= (∂G∂h*∂h∂_ .+ ∂G∂σ*∂σ∂_)[:]
    end
    return nothing
end


function adjust_gradient_θ_transformed!(gradient::Union{AbstractVector, SubArray},
                                        _gradient::AbstractVector,
                                        ∂G∂p::AbstractVector,
                                        θ_dynamic::Vector{Float64},
                                        θ_indices::ParameterIndices,
                                        simulation_condition_id::Symbol;
                                        autodiff_sensitivites::Bool=false,
                                        adjoint::Bool=false)::Nothing

    map_condition_id = θ_indices.maps_conidition_id[simulation_condition_id]
    map_ode_problem = θ_indices.map_ode_problem

    # Transform gradient parameter that for each experimental condition appear in the ODE system
    i_change = θ_indices.map_ode_problem.iθ_dynamic
    if autodiff_sensitivites == true
        gradient1 = _gradient[map_ode_problem.iθ_dynamic] .+ ∂G∂p[map_ode_problem.i_ode_problem_θ_dynamic]
    else
        gradient1 = _gradient[map_ode_problem.i_ode_problem_θ_dynamic] .+ ∂G∂p[map_ode_problem.i_ode_problem_θ_dynamic]
    end
    @views gradient[i_change] .+= _adjust_gradient_θ_transformed(gradient1,
                                                                      θ_dynamic[map_ode_problem.iθ_dynamic],
                                                                      θ_indices.θ_dynamic_names[map_ode_problem.iθ_dynamic],
                                                                      θ_indices)

    # For forward sensitives via autodiff ∂G∂p is on the same scale as ode_problem.p, while
    # S-matrix is on the same scale as θ_dynamic. To be able to handle condition specific
    # parameters mapping to several ode_problem.p parameters the sensitivity matrix part and
    # ∂G∂p must be treated seperately.
    if autodiff_sensitivites == true
        _iθ_dynamic = unique(map_condition_id.iθ_dynamic)
        gradient[_iθ_dynamic] .+= _adjust_gradient_θ_transformed(_gradient[_iθ_dynamic],
                                                                      θ_dynamic[_iθ_dynamic],
                                                                      θ_indices.θ_dynamic_names[_iθ_dynamic],
                                                                      θ_indices)
        out = _adjust_gradient_θ_transformed(∂G∂p[map_condition_id.i_ode_problem_θ_dynamic],
                                                   θ_dynamic[map_condition_id.iθ_dynamic],
                                                   θ_indices.θ_dynamic_names[map_condition_id.iθ_dynamic],
                                                   θ_indices)
        @inbounds for i in eachindex(map_condition_id.i_ode_problem_θ_dynamic)
            gradient[map_condition_id.iθ_dynamic[i]] += out[i]
        end
    end

    # Here both ∂G∂p and _gradient are on the same scale a ode_problem.p. One condition specific parameter
    # can map to several parameters in ode_problem.p
    if adjoint == true || autodiff_sensitivites == false
        out = _adjust_gradient_θ_transformed(_gradient[map_condition_id.i_ode_problem_θ_dynamic] .+ ∂G∂p[map_condition_id.i_ode_problem_θ_dynamic],
                                                   θ_dynamic[map_condition_id.iθ_dynamic],
                                                   θ_indices.θ_dynamic_names[map_condition_id.iθ_dynamic],
                                                   θ_indices)
        @inbounds for i in eachindex(map_condition_id.i_ode_problem_θ_dynamic)
            gradient[map_condition_id.iθ_dynamic[i]] += out[i]
        end
    end

    return nothing
end


function _adjust_gradient_θ_transformed(_gradient::AbstractVector{T},
                                        θ::AbstractVector{T},
                                        n_parameters_estimate::AbstractVector{Symbol},
                                        θ_indices::ParameterIndices)::Vector{T} where T<:Real

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
