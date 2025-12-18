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
p3 = PEtabParameter(:k3; scale = :log, lb = 1e-2, prior = LogNormal(1.0, 1.0))
@test "$p1" == "PEtabParameter: k1 estimated on log10-scale with bounds [1.0e-03, 1.0e+03]"
@test "$p2" == "PEtabParameter: k2 estimated on lin-scale with bounds [1.0e-03, 1.0e+03]"
@test "$p3" == "PEtabParameter: k3 estimated on log-scale with bounds [1.0e-02, 1.0e+03] and prior LogNormal(μ=1.0, σ=1.0)"

t = default_t()
@variables A(t) B(t)
@parameters sigma
obs1 = PEtabObservable(:a, 1.0)
obs2 = PEtabObservable(A + B, sigma)
obs3 = PEtabObservable((A + B)/B, sigma * B; transformation = :log10)
@test "$obs1" == "PEtabObservable: h = a and sd = 1.0 with normal measurement noise"
@test "$obs2" == "PEtabObservable: h = B(t) + A(t) and sd = sigma with normal measurement noise"
@test "$obs3" == "PEtabObservable: h = (B(t) + A(t)) / B(t) and sd = sigma*B(t) with log-normal measurement noise"

t = default_t()
@variables A(t) B(t)
@parameters k1 k2
cond1 = PEtabCondition(:c1, A, 1.0)
cond2 = PEtabCondition(:c1, [:A, :B], [:k1, :k2])
cond3 = PEtabCondition(:c1, ["A", :B], [k1 + k2, "k1 / k2"])
@test "$cond1" == "PEtabCondition: A(t) = 1.0"
@test "$cond2" == "PEtabCondition: [A, B] = [k1, k2]"
@test "$cond3" == "PEtabCondition: [A, B] = [k1 + k2, k1 / k2]"

event1 = PEtabEvent(:k1, [:A, B], [1.0, B + 2])
event2 = PEtabEvent(sigma == t, [A, B], [1.0, B + 2])
event3 = PEtabEvent(A ≤ 4.0, B, B + 2)
@test "$event1" == "PEtabEvent: Condition k1 and assignments [A, B(t)] = [1.0, 2 + B(t)]"
@test "$event2" == "PEtabEvent: Condition sigma == t and assignments [A(t), B(t)] = [1.0, 2 + B(t)]"
@test "$event3" == "PEtabEvent: Condition A(t) <= 4.0 and assignment B(t) = 2 + B(t)"

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
