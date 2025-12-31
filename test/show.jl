using PEtab, Sundials, OrdinaryDiffEqRosenbrock, StyledStrings, NonlinearSolve,
    Distributions, Catalyst, Test

solver1 = ODESolver(Rodas5P())
solver2 = ODESolver(CVODE_BDF(); abstol = 1e-3, reltol = 1e-8, maxiters = 1000)
@test "$solver1" == "ODESolver: Rodas5P with options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1e+04)"
@test "$solver2" == "ODESolver: CVODE_BDF with options (abstol, reltol, maxiters) = (1.0e-03, 1.0e-08, 1e+03)"

ss_solver1 = SteadyStateSolver(:Simulate)
ss_solver2 = SteadyStateSolver(:Simulate; termination_check = :Newton)
ss_solver3 = SteadyStateSolver(:Rootfinding)
@test "$(ss_solver1)" == "SteadyStateSolver: Simulate ODE until du = f(u, p, t) ≈ 0\nTerminates when wrms = (∑((du ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
@test "$(ss_solver2)" == "SteadyStateSolver: Simulate ODE until du = f(u, p, t) ≈ 0\nTerminates when Newton step Δu = √(∑((Δu ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
@test "$(ss_solver3)" == "SteadyStateSolver: Rootfinding to solve du = f(u, p, t) ≈ 0\nAlgorithm: GeneralizedFirstOrderAlgorithm with NonlinearSolve.jl termination"

p1 = PEtabParameter(:k1)
p2 = PEtabParameter(:k2; scale = :lin)
p3 = PEtabParameter(:k3; scale = :log, lb = 1e-2, ub = 1000.0, prior = LogNormal(1.0, 1.0))
p4 = PEtabParameter(:k4; scale = :lin, prior = LogNormal(1.0, 1.0))
p5 = PEtabParameter(:k5; value = 3.0, estimate = false)
@test "$p1" == "PEtabParameter k1: estimate (scale = log10, bounds = [1.0e-03, 1.0e+03])"
@test "$p2" == "PEtabParameter k2: estimate (scale = lin, bounds = [1.0e-03, 1.0e+03])"
@test "$p3" == "PEtabParameter k3: estimate (scale = log, prior(k3) = Truncated(LogNormal(μ=1.0, σ=1.0); lower=0.01, upper=1000.0))"
@test "$p4" == "PEtabParameter k4: estimate (scale = lin, prior(k4) = LogNormal(μ=1.0, σ=1.0))"
@test "$p5" == "PEtabParameter k5: fixed = 3.00e+00"

t = default_t()
@variables A(t) B(t)
@parameters sigma
obs1 = PEtabObservable(:o1, :a, 1.0)
obs2 = PEtabObservable("o1", A + B, sigma)
obs3 = PEtabObservable("o1", A + B, sigma + B)
obs4 = PEtabObservable(:o1, :a, 1.0; distribution = LogNormal)
obs5 = PEtabObservable(:o1, :a, 1.0; distribution = PEtab.Log2Normal)
obs6 = PEtabObservable(:o1, :a, 1.0; distribution = PEtab.Log10Normal)
obs7 = PEtabObservable(:o1, :a, 1.0; distribution = Laplace)
obs8 = PEtabObservable(:o1, :a, 1.0; distribution = PEtab.LogLaplace)
@test "$obs1" == "PEtabObservable o1: data ~ Normal(μ=a, σ=1.0)"
@test "$obs2" == "PEtabObservable o1: data ~ Normal(μ=(B(t) + A(t)), σ=sigma)"
@test "$obs3" == "PEtabObservable o1: data ~ Normal(μ=(B(t) + A(t)), σ=(sigma + B(t)))"
@test "$obs4" == "PEtabObservable o1: log(data) ~ Normal(μ=log(a), σ=1.0)"
@test "$obs5" == "PEtabObservable o1: log2(data) ~ Normal(μ=log2(a), σ=1.0)"
@test "$obs6" == "PEtabObservable o1: log10(data) ~ Normal(μ=log10(a), σ=1.0)"
@test "$obs7" == "PEtabObservable o1: data ~ Laplace(μ=a, θ=1.0)"
@test "$obs8" == "PEtabObservable o1: log(data) ~ Laplace(μ=log(a), θ=1.0)"

t = default_t()
@variables A(t) B(t)
@parameters k1 k2
cond1 = PEtabCondition(:c1, A => 1.0)
cond2 = PEtabCondition(:c1, :A => :k1, :B => :k2)
cond3 = PEtabCondition(:c1, "A" => exp(k1 + k2), :B => "k1 / k2")
cond4 = PEtabCondition(:c1)
@test "$cond1" == "PEtabCondition c1: A(t) => 1.0"
@test "$cond2" == "PEtabCondition c1: A => k1, B => k2"
@test "$cond3" == "PEtabCondition c1: A => exp(k1 + k2), B => k1 / k2"
@test "$cond4" == "PEtabCondition c1:"

event1 = PEtabEvent(:k1, :A => 1.0, B => B + 2)
event2 = PEtabEvent(sigma == t, A => 1.0, B => B + 2)
event3 = PEtabEvent(A ≤ 4.0, B => B + 2)
@test "$event1" == "PEtabEvent when k1: A => 1.0, B(t) => 2 + B(t)"
@test "$event2" == "PEtabEvent when sigma == t: A(t) => 1.0, B(t) => 2 + B(t)"
@test "$event3" == "PEtabEvent when A(t) <= 4.0: B(t) => 2 + B(t)"

path1 = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
path2 = joinpath(@__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
model1 = PEtabModel(path1; build_julia_files = true, verbose = false, write_to_file = false)
model2 = PEtabModel(path2; build_julia_files = true, verbose = false, write_to_file = false)
@test "$model1"[1:66] == "PEtabModel: Boehm_JProteomeRes2014 with 8 states and 10 parameters"
@test "$model2"[1:61] == "PEtabModel: Brannmark_JBC2010 with 9 states and 23 parameters"

ms_res = PEtabMultistartResult(joinpath(@__DIR__, "optimisation_results", "boehm"))
@test "$(ms_res)"[1:188] == "PEtabMultistartResult\n---------------- Summary ---------------\nmin(f)                = 1.38e+02\nParameters estimated  = 9\nNumber of multistarts = 100\nOptimiser algorithm   = Optim_IPNewton"
@test "$(ms_res.runs[1])" == "PEtabOptimisationResult\n---------------- Summary ---------------\nmin(f)                = 1.50e+02\nParameters estimated  = 9\nOptimiser iterations  = 47\nRuntime               = 4.6e+00s\nOptimiser algorithm   = Optim_IPNewton\n"

alg = IpoptOptimizer(false)
@test "$alg" == "Ipopt(LBFGS = false)"

#=
A fun bug in Julia
prob1 = PEtabODEProblem(model1)
prob2 = PEtabODEProblem(model2; verbose = false)
@test prob1) == "PEtabODEProblem: Boehm_JProteomeRes2014 with ODE-states 8 and 9 parameters to estimate\n---------------- Problem options ---------------\nGradient method: ForwardDiff\nHessian method: ForwardDiff\nODE-solver nllh: Rodas5P\nODE-solver gradient: Rodas5P"
@test prob2) == "PEtabODEProblem: Brannmark_JBC2010 with ODE-states 9 and 22 parameters to estimate\n---------------- Problem options ---------------\nGradient method: ForwardDiff\nHessian method: ForwardDiff\nODE-solver nllh: Rodas5P\nODE-solver gradient: Rodas5P\nss-solver: Simulate ODE until du = f(u, p, t) ≈ 0\nss-solver gradient: Simulate ODE until du = f(u, p, t) ≈ 0"
=#
