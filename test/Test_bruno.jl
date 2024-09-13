using Catalyst
using PEtab
using OrdinaryDiffEq
using Test
using Catalyst
using DataFrames
using CSV
using DataFrames
using Statistics
using LinearAlgebra

# Last value pyPESTO :

path_yaml = joinpath(@__DIR__, "published_models", "Bruno_JExpBot2016", "Bruno_JExpBot2016.yaml")

model = PEtabModel(path_yaml)
prob = PEtabODEProblem(model, gradient_method=:ForwardEquations, hessian_method=:GaussNewton,
                       sensealg=:ForwardDiff, ode_solver=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8),
                       split_over_conditions=true)

x = prob.xnominal_transformed
prob.nllh(x)
g1 = prob.grad(x)
H_jl = prob.hess(x)

jlnames = string.(prob.xnames)
pynames = Matrix(CSV.read("/home/sebpe/Dropbox/PhD/Projects/Master-Thesis/Benchmarks/Cost_grad_hess/model_Bruno_JExpBot2016/Names.csv", DataFrame, header=false))[:, 1]

_h_pyPESTO = Matrix(CSV.read("/home/sebpe/Dropbox/PhD/Projects/Master-Thesis/Benchmarks/Cost_grad_hess/model_Bruno_JExpBot2016/Hess.csv", DataFrame, header=false))
h_pyPESTO = zeros(length(g1), length(g1))
g_pyPESTO = Matrix(CSV.read("/home/sebpe/Dropbox/PhD/Projects/Master-Thesis/Benchmarks/Cost_grad_hess/model_Bruno_JExpBot2016/Grad.csv", DataFrame, header=false))[:, 1]
imap = [findfirst(x -> x == jlnames[i], pynames) for i in eachindex(jlnames)]
for (i1, i2) in pairs(imap)
    for (j1, j2) in pairs(imap)
        h_pyPESTO[i1, j1] = _h_pyPESTO[i2, j2]
    end
end

using LinearAlgebra
diff_g = norm(g1 - g_pyPESTO[imap])
diff_g_norm = norm(normalize(g1) - normalize(g_pyPESTO[imap]))

H_jl - h_pyPESTO
norm(H_jl - h_pyPESTO)

path_yaml = joinpath(@__DIR__, "published_models", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
model = PEtabModel(path_yaml, build_julia_files=true)
tollist = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
gradlist = Vector{Vector{Float64}}(undef, length(tollist))
hesslist = Vector{Matrix{Float64}}(undef, length(tollist))
for (i, tol) in pairs(tollist)
    prob = PEtabODEProblem(model, gradient_method=:ForwardEquations, hessian_method=:GaussNewton,
                           sensealg=:ForwardDiff, ode_solver=ODESolver(Rodas5P(), abstol=tol, reltol=tol),
                           verbose=false)
    x = prob.xnominal_transformed
    obj = prob.nllh(x)
    gradlist[i] = prob.grad(x)
    hesslist[i] = prob.hess(x)
end


grad_high = gradlist[end]
hess_high = hesslist[end]
gradnorm = [norm(grad_high - grad) for grad in gradlist]
hessnorm = [norm(hess_high - hess) for hess in hesslist]
compgrad = [maximum(abs.(grad - grad_high)) for grad in gradlist]
comphess = [maximum(abs.(hess - hess_high)) for hess in hesslist]

path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml, build_julia_files=true)
tollist = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
gradlist = Vector{Vector{Float64}}(undef, length(tollist))
hesslist = Vector{Matrix{Float64}}(undef, length(tollist))
for (i, tol) in pairs(tollist)
    prob = PEtabODEProblem(model, gradient_method=:ForwardEquations, hessian_method=:GaussNewton,
                           sensealg=:ForwardDiff, ode_solver=ODESolver(Rodas5P(), abstol=tol, reltol=tol),
                           verbose=false)
    x = prob.xnominal_transformed
    obj = prob.nllh(x)
    gradlist[i] = prob.grad(x)
    hesslist[i] = prob.hess(x)
end

grad_high = gradlist[end]
hess_high = hesslist[end]
gradnorm = [norm(grad_high - grad) for grad in gradlist]
hessnorm = [norm(hess_high - hess) for hess in hesslist]
compgrad = [maximum(abs.(grad - grad_high)) for grad in gradlist]
comphess = [maximum(abs.(hess - hess_high)) for hess in hesslist]

path_yaml = joinpath(@__DIR__, "published_models", "Isensee_JCB2018", "Isensee_JCB2018.yaml")
model = PEtabModel(path_yaml, build_julia_files=true)
tollist = [1e-6, 1e-8, 1e-10, 1e-11]
gradlist = Vector{Vector{Float64}}(undef, length(tollist))
hesslist = Vector{Matrix{Float64}}(undef, length(tollist))
for (i, tol) in pairs(tollist)
    prob = PEtabODEProblem(model, gradient_method=:ForwardEquations, hessian_method=:GaussNewton,
                           sensealg=:ForwardDiff, ode_solver=ODESolver(QNDF(), abstol=tol, reltol=tol),
                           verbose=false) # As in the benchmark
    x = prob.xnominal_transformed
    obj = prob.nllh(x)
    gradlist[i] = prob.grad(x)
    hesslist[i] = prob.hess(x)
end
grad_high = gradlist[end]
hess_high = hesslist[end]
gradnorm = [norm(grad_high - grad) for grad in gradlist]
hessnorm = [norm(hess_high - hess) for hess in hesslist]
compgrad = [maximum(abs.(grad - grad_high)) for grad in gradlist]
comphess = [maximum(abs.(hess - hess_high)) for hess in hesslist]


# -----------------------------------
# MVE
using ModelingToolkit, MethodOfLines, Plots, DomainSets, OrdinaryDiffEq, PEtab, DataFrames

pythonplot()

# Parameters, variables, and derivatives
@parameters k a To
@variables t x T(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

Tamb = 273 #K
To_test = 1000 #K
h = 30 #W/m^2K arbritrary value
L = 0.1 #m
ρ = 7500 #kg/m^3
cp = 420 #J/kgK
kin = 15 #W/m K
ain = kin/ρ/cp #m^2/s

param = [k => kin, a => ain, To => To_test]

# 1D PDE and boundary conditions
eq  = Dt(T(t, x)) ~ a * Dxx(T(t, x))
bcs = [T(0, x) ~ Tamb, #initial condition
        Dx(T(t, 0)) ~ -h/k *(T(t, 0) - Tamb), #cold end
        T(t, L) ~ To] #hot end

# Space and time domains
domains = [t ∈ Interval(0.0, 300.0),
           x ∈ Interval(0.0, L)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [T(t, x)], param)

# Method of lines discretization
dx = L/100
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
sol = solve(prob, Tsit5(), saveat=1.)

# Plot the solution
px = collect(sol[x])
pt = sol[t]
pT = sol[T(t,x)]
plt = plot(sol[t], sol[T(t,x)][:, 1], xlabel="Time", ylabel="Temperature", title="1D Heat Equation")
cnt = contourf(px, pt, pT, xlabel="Position", ylabel="Time", title="1D Heat Equation")
display(cnt)

#extract simulated experiments to create measurements for PEtab
idx =[3, 50, 97]
mx = px[idx]
mT = pT[:, idx]

#Create PEtab problem
odesys = symbolic_discretize(pdesys, discretization)
simpsys = structural_simplify(odesys[1])
@unpack T, k, a, To = simpsys
obs_T3 = PEtabObservable(T[3], 0.5)
observables = Dict("obs_T3" => obs_T3)
_k = PEtabParameter(:k, lb=0.1, ub=100., scale=:lin)
_a = PEtabParameter(:a, lb=1E-7, ub=1E-4, scale=:log10)
params = [_k , _a]
E0 = Dict(:To => 1000.)
E1 = Dict(:To => 900.)
sim_cond = Dict("c0" => E0, "c1" => E1)
measurements = DataFrame(
    simulation_id = ["c0", "c1"],
    obs_id=["obs_T3", "obs_T3"],
    time = [300., 300.],
    measurement=[400., 420.])

model = PEtabModel(simpsys, sim_cond, observables, measurements, params, verbose=true)
petab_problem = PEtabODEProblem(model, verbose=false)


p0 = generate_startguesses(petab_problem, 1)
res = calibrate_model(petab_problem, p0, IpoptOptimiser(false),
                      options=IpoptOptions(max_iter = 1000))
println(res)

solver = Rodas5P()
