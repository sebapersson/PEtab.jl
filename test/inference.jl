using PEtab, OrdinaryDiffEqRosenbrock, ModelingToolkit, Distributions, Random, DataFrames, CSV
using Bijectors, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Catalyst, Test
using AdaptiveMCMC, AdvancedHMC

function get_reference_stats(path_data)
    # Reference chain 10000 samples via Turing of HMC
    chain_reference_df = CSV.read(path_data, DataFrame)
    _chain_reference = Array{Float64, 3}(undef, 10000, 3, 1)
    _chain_reference[:, :, 1] .= Matrix(chain_reference_df)
    chain_reference = MCMCChains.Chains(_chain_reference)
    reference_stats = summarystats(chain_reference)
    return reference_stats
end

function get_prob_saturated(pest)::PEtabODEProblem
    t = default_t()
    D = default_time_deriv()
    @mtkmodel SYS begin
        @parameters begin
            b1
            b2
        end
        @variables begin
            x(t) = 0.0
        end
        @equations begin
            D(x) ~ b2*(b1 - x)
        end
    end
    @mtkbuild sys = SYS()

    Random.seed!(1234)
    # Simulate the model
    parametermap = [:b1 => 1.0, :b2 => 0.2]
    oprob = ODEProblem(sys, [], (0.0, 2.5), parametermap)
    tsave = collect(range(0.0, 2.5, 101))
    dist = Normal(0.0, 0.03)
    _sol = solve(oprob, Rodas4(), abstol=1e-12, reltol=1e-12, saveat=tsave, tstops=tsave)
    obs = _sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))

    ## Setup the parameter estimation problem
    @parameters sigma
    observables = PEtabObservable("obs_X", :x, sigma)

    measurements = DataFrame(
        obs_id="obs_X",
        time=_sol.t,
        measurement=obs)

    model = PEtabModel(sys, observables, measurements, pest; verbose = false)
    return PEtabODEProblem(model; odesolver=ODESolver(Rodas5(), abstol=1e-6, reltol=1e-6))
end

@testset "Check inference linear priors + parameters" begin
    _b1 = PEtabParameter(:b1, value=1.0, lb=0.0, ub=5.0, scale=:lin)
    _b2 = PEtabParameter(:b2, value=0.2, lb=0.0, ub=5.0, scale=:lin)
    _sigma = PEtabParameter(:sigma, value=0.03, lb=1e-3, ub=1e2, scale=:lin)
    pest = [_b1, _b2, _sigma]
    prob = get_prob_saturated(pest)
    # Reference chain based on 10,000 iterations
    reference_stats = get_reference_stats(joinpath(@__DIR__, "inference_results", "Saturated_chain.csv"))

    # HMC inference case
    Random.seed!(123)
    target = PEtabLogDensity(prob)
    sampler = NUTS(0.8)
    xprior = to_prior_scale(prob.xnominal_transformed, target)
    xinference = target.inference_info.bijectors(xprior)
    res = sample(target, sampler,
                 3000;
                 n_adapts = 1000,
                 initial_params = xinference,
                 drop_warmup=true,
                 progress=false,
                 verbose=true)
    chain_hmc = to_chains(res, target)
    hmc_stats = summarystats(chain_hmc)
    @testset "HMC" begin
        @test reference_stats.nt.mean[1] ≈ hmc_stats.nt.mean[1] atol=2e-1
        @test reference_stats.nt.mean[2] ≈ hmc_stats.nt.mean[2] atol=1e-2
        @test reference_stats.nt.mean[3] ≈ hmc_stats.nt.mean[3] atol=1e-2
        @test reference_stats.nt.std[1] ≈ hmc_stats.nt.std[1] atol=1e-1
        @test reference_stats.nt.std[2] ≈ hmc_stats.nt.std[2] atol=1e-2
        @test reference_stats.nt.std[3] ≈ hmc_stats.nt.std[3] atol=1e-2
    end

    # AdaptiveMCMC
    Random.seed!(1234)
    target = PEtabLogDensity(prob)
    res = adaptive_rwm(xinference, target.logtarget, 200000; progress=false)
    chain_adapt = to_chains(res, target)
    adaptive_stats = summarystats(chain_adapt)
    @testset "Adaptive MCMC" begin
        @test reference_stats.nt.mean[1] ≈ adaptive_stats.nt.mean[1] atol=2e-1
        @test reference_stats.nt.mean[2] ≈ adaptive_stats.nt.mean[2] atol=1e-2
        @test reference_stats.nt.mean[3] ≈ adaptive_stats.nt.mean[3] atol=1e-2
        @test reference_stats.nt.std[1] ≈ adaptive_stats.nt.std[1] atol=5e-1
        @test reference_stats.nt.std[2] ≈ adaptive_stats.nt.std[2] atol=1e-2
        @test reference_stats.nt.std[3] ≈ adaptive_stats.nt.std[3] atol=1e-2
    end
end
