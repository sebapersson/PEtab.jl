module PEtabPigeonsExtension

using ModelingToolkit
using Distributions
using PEtab
using Pigeons
using Bijectors
using LogDensityProblems
using LogDensityProblemsAD
using MCMCChains

function Pigeons.initialization(log_potential::PEtab.PEtabLogDensity, rng, ::Int)
    return deepcopy(log_potential.initial_value)
end

function LogDensityProblemsAD.ADgradient(::Symbol, log_potential::PEtab.PEtabLogDensity,
                                         buffers::Pigeons.Augmentation)
    Pigeons.BufferedAD(log_potential, buffers)
end

function LogDensityProblems.logdensity_and_gradient(logpotential::Pigeons.BufferedAD{<:PEtabLogDensity},
                                                    x)
    logdens, grad = logpotential.enclosed.logtarget_gradient(x)
    logpotential.buffer .= grad
    return logdens, logpotential.buffer
end

function Pigeons.sample_iid!(log_prior::PEtab.PEtabPigeonReference, replica, shared)
    @unpack state, rng = replica
    sample_iid!(state, rng, log_prior.inference_info)
end

function sample_iid!(state::AbstractVector, rng,
                     inference_info::PEtab.InferenceInfo)::Nothing
    for i in eachindex(state)
        state[i] = rand(rng, inference_info.priors[i])
    end
    state .= inference_info.bijectors(state)
    return nothing
end

function PEtab.to_chains(res::Pigeons.PT, target::PEtab.PEtabLogDensity;
                         start_time = nothing, end_time = nothing)
    # Dependent on method
    inference_info = target.inference_info
    out = sample_array(res)[:, 1:(end - 1), :]
    for i in 1:size(out)[1]
        out[i, :, :] .= inference_info.inv_bijectors(out[i, :, 1])
    end
    if isnothing(start_time) || isnothing(end_time)
        return MCMCChains.Chains(out, inference_info.parameters_id)
    else
        _chain = MCMCChains.Chains(out, inference_info.parameters_id)
        return MCMCChains.setinfo(_chain, (start_time = start_time, stop_time = end_time))
    end
end

end
