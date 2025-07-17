test_case = "011"

nn11_1 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nn11_2 = @compact(
    layer1 = Dense(2, 5, Lux.relu),
    layer2 = Dense(5, 10, Lux.relu),
    layer3 = Dense(10, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_models = Dict(:net1 => MLModel(nn11_1; static = false),
                 :net2 => MLModel(nn11_2; static = true, inputs = [:net2_input_pre1, :net2_input_pre1], outputs = [:beta]))
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net1_ps.hdf5")
pnn1 = Lux.initialparameters(rng, nn11_1) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn1, path_h5, nn11_1, :net1)
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net2_ps.hdf5")
pnn2 = Lux.initialparameters(rng, nn11_2) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn2, path_h5, nn11_2, :net2)

function _lv11!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p

    net1 = ml_models[:net1]
    du1_nn, st = net1.model([prey, predator], p[:net1], net1.st)
    net1.st = st

    du[1] = alpha*prey - beta * prey * predator # prey
    du[2] = du1_nn[1] - delta*predator # predator
    return nothing
end

lv11! = let _ml_models = ml_models
    (du, u, p, t) -> _lv11!(du, u, p, t, _ml_models)
end
p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
p_model = ComponentArray(merge(p_mechanistic, (net1=pnn1,)))
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = ODEProblem(lv11!, u0, (0.0, 10.0), p_model)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:net2_input_pre1; scale = :lin, lb = 0.0, ub = 15.0, value = 2.0, estimate = false)
p_input2 = PEtabParameter(:net2_input_pre2; scale = :lin, lb = 0.0, ub = 15.0, value = 2.0, estimate = false)
p_net1 = PEtabMLParameter(:net1, true, pnn1)
p_net2 = PEtabMLParameter(:net2, true, pnn2)
pest = [p_alpha, p_delta, p_input1, p_input2, p_net1, p_net2]

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
