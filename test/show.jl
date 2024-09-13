using PEtab, Sundials, OrdinaryDiffEq, StyledStrings, Printf, NonlinearSolve,
      Distributions, Catalyst, Test

solver1 = ODESolver(Rodas5P())
solver2 = ODESolver(CVODE_BDF(); abstol = 1e-3, reltol = 1e-8, maxiters = 1000)
solver3 = ODESolver(Vern7())
@test @sprintf("%s", solver1) == "ODESolver: Rodas5P with options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1e+04)"
@test @sprintf("%s", solver2) == "ODESolver: CVODE_BDF with options (abstol, reltol, maxiters) = (1.0e-03, 1.0e-08, 1e+03)"
@test @sprintf("%s", solver3) == "ODESolver: Vern7 with options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1e+04)"

ss_solver1 = SteadyStateSolver(:Simulate)
ss_solver2 = SteadyStateSolver(:Simulate; termination_check = :Newton)
ss_solver3 = SteadyStateSolver(:Rootfinding)
@test @sprintf("%s", ss_solver1) == "SteadyStateSolver: Simulate ODE until du = f(u, p, t) ≈ 0\nTerminates when wrms = (∑((du ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
@test @sprintf("%s", ss_solver2) == "SteadyStateSolver: Simulate ODE until du = f(u, p, t) ≈ 0\nTerminates when Newton step Δu = √(∑((Δu ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
@test @sprintf("%s", ss_solver3) == "SteadyStateSolver: Rootfinding to solve du = f(u, p, t) ≈ 0\nAlgorithm: TrustRegion with NonlinearSolve.jl termination"

p1 = PEtabParameter(:k1)
p2 = PEtabParameter(:k2; scale = :lin)
p3 = PEtabParameter(:k3; scale = :log, lb = 1e-2, prior = LogNormal(1.0, 1.0))
@test @sprintf("%s", p1) == "PEtabParameter: k1 estimated on log10-scale with bounds [1.0e-03, 1.0e+03]"
@test @sprintf("%s", p2) == "PEtabParameter: k2 estimated on lin-scale with bounds [1.0e-03, 1.0e+03]"
@test @sprintf("%s", p3) == "PEtabParameter: k3 estimated on log-scale with bounds [1.0e-02, 1.0e+03] and prior LogNormal(μ=1.0, σ=1.0)"

t = default_t()
@variables A(t) B(t)
@parameters sigma
obs1 = PEtabObservable(:a, 1.0)
obs2 = PEtabObservable(A + B, sigma)
obs3 = PEtabObservable((A + B)/B, sigma * B; transformation = :log10)
@test @sprintf("%s", obs1) == "PEtabObservable: h = a and sd = 1.0 with normal measurement noise"
@test @sprintf("%s", obs2) == "PEtabObservable: h = B(t) + A(t) and sd = sigma with normal measurement noise"
@test @sprintf("%s", obs3) == "PEtabObservable: h = (B(t) + A(t)) / B(t) and sd = sigma*B(t) with log-normal measurement noise"

event1 = PEtabEvent(:k1, [1.0, B + 2], [:A, B])
event2 = PEtabEvent(sigma == t, [1.0, B + 2], [A, B])
event3 = PEtabEvent(A ≤ 4.0, B + 2, B)
@test @sprintf("%s", event1) == "PEtabEvent: Condition k1 and affect [A, B(t)] = [1.0, 2 + B(t)]"
@test @sprintf("%s", event2) == "PEtabEvent: Condition sigma == t and affect [A(t), B(t)] = [1.0, 2 + B(t)]"
@test @sprintf("%s", event3) == "PEtabEvent: Condition A(t) <= 4.0 and affect B(t) = 2 + B(t)"

path1 = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
path2 = joinpath(@__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
model1 = PEtabModel(path1; build_julia_files = true, verbose = false, write_to_file = false)
model2 = PEtabModel(path2; build_julia_files = true, verbose = false, write_to_file = false)
@test @sprintf("%s", model1)[1:66] == "PEtabModel: Boehm_JProteomeRes2014 with 8 states and 10 parameters"
@test @sprintf("%s", model2)[1:61] == "PEtabModel: Brannmark_JBC2010 with 9 states and 23 parameters"

prob1 = PEtabODEProblem(model1; verbose = false)
prob2 = PEtabODEProblem(model2; verbose = false)
@test @sprintf("%s", prob1) == "PEtabODEProblem: Boehm_JProteomeRes2014 with ODE-states 8 and 9 parameters to estimate\n---------------- Problem options ---------------\nGradient method: ForwardDiff\nHessian method: ForwardDiff\nODE-solver nllh: Rodas5P\nODE-solver gradient: Rodas5P"
@test @sprintf("%s", prob2) == "PEtabODEProblem: Brannmark_JBC2010 with ODE-states 9 and 22 parameters to estimate\n---------------- Problem options ---------------\nGradient method: ForwardDiff\nHessian method: ForwardDiff\nODE-solver nllh: Rodas5P\nODE-solver gradient: Rodas5P\nss-solver: Simulate ODE until du = f(u, p, t) ≈ 0\nss-solver gradient: Simulate ODE until du = f(u, p, t) ≈ 0"

@sprintf("%s", prob2)
