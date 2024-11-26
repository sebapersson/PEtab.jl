test_case = "002"

nn2 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
pnn = Lux.initialparameters(rng, nn2)
nnmodels = Dict(:net1 => NNModel(nn2))

function _lv!(du, u, p, t, nn)
    prey, predator = u
    @unpack alpha, beta, p_net1 = p

    net1 = nnmodels[:net1]
    du_nn, st = net1.nn([prey, predator], p_net1, net1.st)
    net1.st = st

    du[1] = alpha*prey - beta * prey * predator # prey
    du[2] = du_nn[1]
    return nothing
end

lv2! = let nnmodels = nnmodels
    (du, u, p, t) -> _lv!(du, u, p, t, nnmodels)
end
p_mechanistic = (alpha = 1.3, beta = 0.9)
p_model = ComponentArray(merge(p_mechanistic, (p_net1=pnn,)))
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = ODEProblem(lv2!, u0, (0.0, 10.0), p_model)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_net1 = PEtabParameter(:p_net1; scale = :lin, lb = -15.0, ub = 15.0, value = 0.0)
pest = [p_alpha, p_beta, p_net1]

obs_prey = PEtabObservable(:prey, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey" => obs_prey, "predator" => obs_predator)

conds = Dict("cond1" => Dict{Symbol, Symbol}())

path_m = joinpath(@__DIR__, "test_cases", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; nnmodels = nnmodels,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
for config in PROB_CONFIGS
    petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = config.grad,
                                 split_over_conditions = config.split, sensealg=config.sensealg)
    test_model(test_case, petab_prob)
end
