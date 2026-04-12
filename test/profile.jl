using Catalyst, DataFrames, Distributions, LikelihoodProfiler, OptimizationLBFGSB, Optim,
    OrdinaryDiffEqRosenbrock, PEtab, StableRNGs, Test

# Create a PEtab problem to profile
rn = @reaction_network begin
    @parameters S0 c3 = 3.0
    @species begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.0
        P(t) = 0.0
    end
    @observables begin
        obs1 ~ S + E
        obs2 ~ P
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

@parameters sigma
observables = [
    PEtabObservable(:petab_obs1, :obs1, 3.0),
    PEtabObservable(:petab_obs2, :obs2, sigma),
]
pest = [
    PEtabParameter(:c1),
    PEtabParameter(:c2; prior = LogNormal(2.5, 0.3)),
    PEtabParameter(:S0),
    PEtabParameter(:sigma),
]
# Measurements; simulate with 'true' parameters
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs1 = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs2 = sol[:P] .+ randn(length(sol[:P]))
df1 = DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1)
df2 = DataFrame(obs_id = "petab_obs2", time = sol.t, measurement = obs2)
measurements = vcat(df1, df2)

model_rn = PEtabModel(rn, observables, measurements, pest)
petab_prob = PEtabODEProblem(model_rn)

rng = StableRNG(42)
x0 = get_startguesses(rng, petab_prob, 1)
res = calibrate(petab_prob, x0, Optim.IPNewton())

# Profile
pl_prob = ProfileLikelihoodProblem(res, petab_prob)
meth_opt = OptimizationProfiler(
    optimizer = LBFGSB(), stepper = FixedStep(; initial_step = 0.005)
)
pl_sol = solve(pl_prob, meth_opt)

@test pl_sol[1].endpoints[1] > -0.64
@test pl_sol[1].endpoints[2] < 0.07
@test pl_sol[2].endpoints[1] > 0.6
@test pl_sol[2].endpoints[2] < 1.39
@test pl_sol[3].endpoints[1] > 1.94
@test pl_sol[3].endpoints[2] < 2.03
@test pl_sol[4].endpoints[1] > 0.96
@test pl_sol[4].endpoints[2] < 1.42
