test_case = "029"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn29 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do (x1, x2)
    x = cat(x1, x2; dims = 1)
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_models = MLModels(MLModel(:net5, nn29, false))
path_h5 = joinpath(dir_case, "net5_ps.hdf5")
pnn = Lux.initialparameters(rng, nn5) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn5, :net5)

function _lv29!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p
    net5 = ml_models[:net5]
    du_nn, st = net5.lux_model(([prey], [predator]), p[:net5], net5.st)
    net5.st = st

    du[1] = alpha*prey - beta * prey * predator # prey
    du[2] = du_nn[1] - delta*predator # predator
    return nothing
end
lv29! = let _ml_models = ml_models
    (du, u, p, t) -> _lv29!(du, u, p, t, _ml_models)
end

p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
p_ode = ComponentArray(merge(p_mechanistic, (net5=pnn,)))
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = ODEProblem(lv29!, u0, (0.0, 10.0), p_ode)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_net5 = PEtabMLParameter(:net5, true, pnn)
pest = [p_alpha, p_beta, p_delta, p_net5]

observables = [
    PEtabObservable(:prey_o, :prey, 0.05),
    PEtabObservable(:predator_o, :predator, 0.05)
]

conditions = PEtabCondition(:e1)

path_m = joinpath(dir_case, "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)
rename!(measurements, "experimentId" => "simulation_id")

model = PEtabModel(
    uprob, observables, measurements, pest; ml_models = ml_models,
    simulation_conditions = conditions
)
petab_prob = PEtabODEProblem(
    model; odesolver = ode_solver, gradient_method = :ForwardDiff,
    split_over_conditions = true
)
test_hybrid(test_case, petab_prob)
