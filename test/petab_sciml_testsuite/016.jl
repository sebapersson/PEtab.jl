test_case = "016"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn16 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_model = MLModel(:net1, nn16, false)
path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn = Lux.initialparameters(rng, nn16) |> ComponentArray |> f64
PEtab._set_ml_model_ps!(pnn, path_h5, nn16, :net1)

function lv16!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p
    net1 = ml_models[:net1]
    du_nn, st = net1.lux_model([prey, predator], p[:net1], net1.st)
    net1.st = st

    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = du_nn[1] - delta * predator # predator
    return nothing
end

p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = UDEProblem(lv16!, u0, (0.0, 10.0), p_mechanistic, ml_model)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_net1 = PEtabMLParameter(:net1; value = pnn, estimate = false)
pest = [p_alpha, p_beta, p_delta, p_net1]

observables = [
    PEtabObservable(:prey_o, :prey, 0.05),
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
