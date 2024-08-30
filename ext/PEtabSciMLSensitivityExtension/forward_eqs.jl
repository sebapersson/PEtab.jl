#=
    Functions for computing forward-sensitivities with SciMLSensitivity
=#

function PEtab._get_odeproblem_gradient(odeproblem::ODEProblem, gradient_method::Symbol,
                                        sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm)::ODEProblem
    return ODEForwardSensitivityProblem(odeproblem.f, odeproblem.u0, odeproblem.tspan,
                                        odeproblem.p, sensealg = sensealg)
end

function PEtab.solve_sensitivites!(model_info::PEtab.ModelInfo, _solve_conditions!::Function, xdynamic::Vector{<:AbstractFloat}, sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm, probleminfo::PEtab.PEtabODEProblemInfo, cids::Vector{Symbol}, cfg::Nothing, isremade::Bool = false)::Bool
    success = _solve_conditions!(xdynamic, cids)
    return success
end

function PEtab._grad_forward_eqs_cond!(grad::Vector{T}, xdynamic::Vector{T}, xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic::Vector{T}, icid::Int64, sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm, probleminfo::PEtab.PEtabODEProblemInfo, model_info::PEtab.ModelInfo)::Nothing where T <: AbstractFloat
    @unpack θ_indices, simulation_info, petab_model = model_info
    @unpack parameter_info, measurement_info = model_info
    @unpack imeasurements_t, tsaves = simulation_info
    cache = probleminfo.cache

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for the gradient functions
    compute∂G∂u! = (out, u, p, t, i) -> begin
        PEtab.compute∂G∂_(out, u, p, t, i, imeasurements_t[cid], measurement_info,
                          parameter_info, θ_indices, petab_model, xnoise, xobservable,
                          xnondynamic, cache.∂h∂u, cache.∂σ∂u, compute∂G∂U = true)
    end
    compute∂G∂p! = (out, u, p, t, i) -> begin
        PEtab.compute∂G∂_(out, u, p, t, i, imeasurements_t[cid], measurement_info,
                          parameter_info, θ_indices, petab_model, xnoise, xobservable,
                          xnondynamic, cache.∂h∂p, cache.∂σ∂p, compute∂G∂U = false)
    end

    p = sol.prob.p
    ∂G∂p, ∂G∂p_ = zeros(Float64, length(p)), zeros(Float64, length(p))
    ∂G∂u = zeros(Float64, length(states(petab_model.sys_mutated)))
    _grad = zeros(Float64, length(p))
    for (it, tsave) in pairs(tsaves[cid])
        u, _S = extract_local_sensitivities(sol, it, true)
        compute∂G∂u!(∂G∂u, u, p, tsave, it)
        compute∂G∂p!(∂G∂p_, u, p, tsave, it)
        _grad .+= transpose(_S) * ∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Adjust if gradient is non-linear scale (e.g. log and log10). TODO: Refactor
    # this function later
    PEtab.adjust_gradient_θ_transformed!(grad, _grad, ∂G∂p, xdynamic, θ_indices,
                                         simid,
                                         autodiff_sensitivites = false)
    return nothing
end
