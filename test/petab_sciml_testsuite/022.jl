test_case = "022"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn22 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 2)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_model = MLModel(
    :net4, nn22, false; inputs = [:prey, :predator], outputs = [:net4_output1, :net4_output2]
)
path_h5 = joinpath(dir_case, "net4_ps.hdf5")
pnn = Lux.initialparameters(rng, nn22) |> ComponentArray |> f64
PEtab._set_ml_model_ps!(pnn, path_h5, nn22, :net4)

function lv22!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p
    net1 = ml_models[:net4]
    du_nn, st = net1.lux_model([prey, predator], p[:net4], net1.st)
    net1.st = st

    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = du_nn[2] - delta * predator # predator
    return nothing
end

p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
u0 = (prey = 0.44249296, predator = 4.6280594)
uprob = UDEProblem(lv22!, u0, (0.0, 10.0), p_mechanistic, ml_model)

pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
    PEtabMLParameter(:net4; value = pnn)
]

observables = [
    PEtabObservable(:prey_o, :net4_output1, 0.05),
    PEtabObservable(:predator_o, :predator, 0.05),
]

conditions = PEtabCondition(:e1)

path_m = joinpath(dir_case, "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)
rename!(measurements, "experimentId" => "simulation_id")

model = PEtabModel(
    uprob, observables, measurements, pest; ml_models = ml_model,
    simulation_conditions = conditions
)
petab_prob = PEtabODEProblem(
    model; odesolver = ode_solver, gradient_method = :ForwardDiff,
    split_over_conditions = true
)
test_hybrid(test_case, petab_prob)
