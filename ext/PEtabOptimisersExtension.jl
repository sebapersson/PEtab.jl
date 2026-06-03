module PEtabOptimisersExtension

import Dates
import Optimisers
import QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm
import Random
using ComponentArrays: ComponentArray
import PEtab

function PEtab.calibrate_multistart(
        rng::Random.AbstractRNG, prob::PEtab.PEtabODEProblem, alg::Optimisers.AbstractRule,
        nmultistarts; nprocs = 1, save_trace = false, dirsave = nothing,
        sample_prior = true, sampling_method = LatinHypercubeSample(),
        init_weight = nothing, init_bias = nothing,
        options::PEtab.OptimisersOptions = PEtab.OptimisersOptions()
    )::PEtab.PEtabMultistartResult
    return PEtab._calibrate_multistart(
        rng, prob, alg, nmultistarts, dirsave, sampling_method, options, sample_prior,
        save_trace, nprocs, init_weight, init_bias
    )
end

function PEtab.calibrate(
        prob::PEtab.PEtabODEProblem, x::Union{Vector{<:AbstractFloat}, ComponentArray},
        alg::Optimisers.AbstractRule; save_trace::Bool = false,
        options::PEtab.OptimisersOptions = PEtab.OptimisersOptions()
    )::PEtab.PEtabOptimisationResult
    iterations = options.iterations

    ftrace = Vector{Float64}(undef, 0)
    xtrace = Vector{Vector{Float64}}(undef, 0)
    epoch_tracker = zeros(Int64, 1)

    converged = :not_applicable
    x0 = deepcopy(x)
    x_prev = deepcopy(x)
    state = Optimisers.setup(alg, x)

    start_time = Dates.now()
    runtime = @elapsed begin
        for epoch in 1:iterations
            epoch_tracker[1] = epoch

            g = prob.grad(x)
            state, x = Optimisers.update(state, x, g)

            # Check if should terminate
            nllh = prob.nllh(x)
            current_time = Dates.now() - start_time
            @assert current_time isa Dates.Millisecond "Time diff. must be in milliseconds"
            if all(x .== x_prev)
                @warn "In epoch $epoch parameter vector x was unchanged. This suggests \
                    gradient computation error; terminating early."
                break
            end
            if isinf(nllh)
                @warn "Objective is infinite in epoch $epoch; terminating early."
                break
            end
            if current_time.value / 1.0e3 > options.max_time
                @warn "Maximum time of $(options.max_time) seconds exceeded; terminating \
                    early."
                break
            end

            if save_trace
                nllh = prob.nllh(x)
                push!(ftrace, nllh)
                push!(xtrace, x)
            end
            copy!(x_prev, x)
        end
    end

    x0_out = PEtab._get_x_out(x0, prob)
    xmin_out = PEtab._get_x_out(x, prob)
    nllh = prob.nllh(x)
    fmin = isinf(nllh) ? NaN : nllh

    ix = findfirst(x -> x == '(', string(alg))
    alg_used = Symbol(string(alg)[1:ix - 1])
    niterations = epoch_tracker[1]
    res = state
    return PEtab.PEtabOptimisationResult(
        xmin_out, fmin, x0_out, alg_used, niterations, runtime, xtrace, ftrace, converged,
        res
    )
end

end
