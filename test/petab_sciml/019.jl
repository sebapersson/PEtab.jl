test_case = "019"

nn19 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 2)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_models = Dict(:net4 => MLModel(nn19; static = false))
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net4_ps.hdf5")
pnn = Lux.initialparameters(rng, nn19) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn19, :net4)

function _lv19!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p
    net1 = ml_models[:net4]
    du_nn, st = net1.model([prey, predator], p[:net4], net1.st)
    net1.st = st

    du[1] = du_nn[1] - beta * prey * predator # prey
    du[2] = du_nn[2] - delta*predator # predator
    return nothing
end
lv19! = let _ml_models = ml_models
    (du, u, p, t) -> _lv19!(du, u, p, t, _ml_models)
end

p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
p_ode = ComponentArray(merge(p_mechanistic, (net4=pnn,)))
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = ODEProblem(lv19!, u0, (0.0, 10.0), p_ode)

p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_net4 = PEtabMLParameter(:net4, true, pnn)
pest = [p_beta, p_delta, p_net4]

obs_prey = PEtabObservable(:prey, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey_o" => obs_prey, "predator_o" => obs_predator)

conds = Dict("cond1" => Dict{Symbol, Symbol}())

path_m = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; ml_models = ml_models,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = :ForwardDiff,
                             split_over_conditions = true)
test_hybrid(test_case, petab_prob)
