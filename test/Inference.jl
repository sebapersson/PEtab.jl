using PEtab, OrdinaryDiffEq, ModelingToolkit, Distributions, Random, DataFrames, CSV
using Bijectors, LogDensityProblems, LogDensityProblemsAD, MCMCChains
using AdaptiveMCMC, AdvancedHMC
using Pigeons

function get_reference_stats(path_data)
    # Reference chain 10000 samples via Turing of HMC
    chain_reference_df = CSV.read(path_data, DataFrame)
    _chain_reference = Array{Float64, 3}(undef, 10000, 3, 1)
    _chain_reference[:, :, 1] .= Matrix(chain_reference_df)
    chain_reference = MCMCChains.Chains(_chain_reference)
    reference_stats = summarystats(chain_reference)
    return reference_stats
end

function get_petab_problem_saturated(parameters)::PEtabODEProblem

    @parameters b1, b2
    @variables t x(t)
    D = Differential(t)
    eqs = [
        D(x) ~ b2*(b1 - x)
    ]
    specie_map = [x => 0]
    parametermap = [b1 => 1.0, b2 => 0.2]
    @named sys = ODESystem(eqs)

    Random.seed!(1234)
    # Simulate the model
    oprob = ODEProblem(sys, specie_map, (0.0, 2.5), parametermap)
    tsave = collect(range(0.0, 2.5, 101))
    dist = Normal(0.0, 0.03)
    _sol = solve(oprob, Rodas4(), abstol=1e-12, reltol=1e-12, saveat=tsave, tstops=tsave)
    obs = _sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))

    ## Setup the parameter estimation problem
    @parameters sigma
    obs_X = PEtabObservable(x, sigma)
    observables = Dict("obs_X" => obs_X)

    measurements = DataFrame(
        obs_id="obs_X",
        time=_sol.t,
        measurement=obs)

    model = PEtabModel(sys, observables, measurements, parameters; statemap=specie_map)
    petab_problem = PEtabODEProblem(model; ode_solver=ODESolver(Rodas5(), abstol=1e-6, reltol=1e-6))

    return petab_problem
end

@testset "Check inference linear priors + parameters" begin

    @parameters b1 b2 sigma
    _b1 = PEtabParameter(b1, value=1.0, lb=0.0, ub=5.0, scale=:lin)
    _b2 = PEtabParameter(b2, value=0.2, lb=0.0, ub=5.0, scale=:lin)
    _sigma = PEtabParameter(sigma, value=0.03, lb=1e-3, ub=1e2, scale=:lin)
    pest = [_b1, _b2, _sigma]
    petab_problem = get_petab_problem_saturated(pest)
    # Reference chain based on 10,000 iterations
    reference_stats = get_reference_stats(joinpath(@__DIR__, "Bayesian_data", "Saturated_chain.csv"))

    # HMC inference case
    Random.seed!(1213)
    target = PEtabLogDensity(petab_problem)
    sampler = NUTS(0.8)
    xprior = to_prior_scale(petab_problem.xnominal_transformed, target)
    xinference = target.inference_info.bijectors(xprior)
    res = sample(target, sampler,
                2000;
                nadapts = 1000,
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
    target = PEtabLogDensity(petab_problem)
    res = adaptive_rwm(xinference, target.logtarget, 200000; progress=true)
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

    # Parallell tempering (with AutoMALA the slowest)
    # Setup with Pigeon.jl
    Random.seed!(123)
    log_potential = PEtabLogDensity(petab_problem)
    log_potential.initial_value .= xinference
    reference_potential = PEtabPigeonReference(petab_problem)
    pt = pigeons(target = log_potential,
                 reference = reference_potential,
                 n_rounds=10,
                 record = [traces; record_default()])
    pt_chain = to_chains(pt, log_potential)
    pt_stats = summarystats(pt_chain)
    @testset "Parallell tempering" begin
        @test reference_stats.nt.mean[1] ≈ pt_stats.nt.mean[1] atol=2e-1
        @test reference_stats.nt.mean[2] ≈ pt_stats.nt.mean[2] atol=2e-1
        @test reference_stats.nt.mean[3] ≈ pt_stats.nt.mean[3] atol=1e-2
        @test reference_stats.nt.std[1] ≈ pt_stats.nt.std[1] atol=5e-1
        @test reference_stats.nt.std[2] ≈ pt_stats.nt.std[2] atol=1e-2
        @test reference_stats.nt.std[3] ≈ pt_stats.nt.std[3] atol=1e-2
    end

    # Check AutoMALA runs
    Random.seed!(123)
    pt = pigeons(target = log_potential,
                 reference = reference_potential,
                 explorer=AutoMALA(),
                 n_rounds=6,
                 record = [traces; record_default()])
    pt_chain = to_chains(pt, log_potential)
    pt_stats = summarystats(pt_chain)
    @test reference_stats.nt.mean[3] ≈ pt_stats.nt.mean[3] atol=1e-2
end

@testset "Check inference transformed parameters" begin
    # Try mixing
    @parameters b1 b2 sigma
    _b1 = PEtabParameter(b1, value=1.0, lb=0.0, ub=5.0, scale=:log10, prior_on_linear_scale=true, prior=Uniform(0.0, 5.0))
    _b2 = PEtabParameter(b2, value=0.2, lb=0.0, ub=5.0, scale=:log, prior=Uniform(-3, log(5.0)), prior_on_linear_scale=false)
    _sigma = PEtabParameter(sigma, value=0.03, lb=1e-3, ub=1e2, scale=:log10, prior=Uniform(-3, 2.0), prior_on_linear_scale=false)
    pest = [_b1, _b2, _sigma]
    petab_problem = get_petab_problem_saturated(pest)

    reference_stats = get_reference_stats(joinpath(@__DIR__, "Bayesian_data", "Saturated_chain_tparam.csv"))

    Random.seed!(1234)
    target = PEtabLogDensity(petab_problem)
    sampler = NUTS(0.8)
    xprior = to_prior_scale(petab_problem.xnominal_transformed, target)
    xinference = target.inference_info.bijectors(xprior)
    res = sample(target, sampler,
                2000;
                nadapts = 1000,
                initial_params = xinference,
                drop_warmup=true,
                progress=false,
                verbose=true)
    chain_hmc = to_chains(res, target)
    hmc_stats = summarystats(chain_hmc)
    @testset "HMC" begin
        @test reference_stats.nt.mean[1] ≈ hmc_stats.nt.mean[1] atol=5e-2
        @test reference_stats.nt.mean[2] ≈ hmc_stats.nt.mean[2] atol=5e-2
        @test reference_stats.nt.mean[3] ≈ hmc_stats.nt.mean[3] atol=1e-2
        @test reference_stats.nt.std[1] ≈ hmc_stats.nt.std[1] atol=1e-1
        @test reference_stats.nt.std[2] ≈ hmc_stats.nt.std[2] atol=5e-2
        @test reference_stats.nt.std[3] ≈ hmc_stats.nt.std[3] atol=1e-2
    end

    # AdaptiveMCMC
    Random.seed!(1234)
    target = PEtabLogDensity(petab_problem)
    res = adaptive_rwm(xinference, target.logtarget, 200000; progress=true)
    chain_adapt = to_chains(res, target)
    adaptive_stats = summarystats(chain_adapt)
    @testset "Adaptive MCMC" begin
        @test reference_stats.nt.mean[1] ≈ adaptive_stats.nt.mean[1] atol=2e-1
        @test reference_stats.nt.mean[2] ≈ adaptive_stats.nt.mean[2] atol=5e-2
        @test reference_stats.nt.mean[3] ≈ adaptive_stats.nt.mean[3] atol=1e-2
        @test reference_stats.nt.std[1] ≈ adaptive_stats.nt.std[1] atol=5e-1
        @test reference_stats.nt.std[2] ≈ adaptive_stats.nt.std[2] atol=5e-2
        @test reference_stats.nt.std[3] ≈ adaptive_stats.nt.std[3] atol=1e-2
    end
end
