test_case = "030"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn30 = @compact(
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

ml_models = MLModel(
    :net5, nn30, true; inputs = ([:net5_input1], [:net5_input2]), outputs = [:gamma]
)
path_h5 = joinpath(dir_case, "net5_ps.hdf5")
pnn = Lux.initialparameters(rng, nn5) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn5, :net5)

function lv30!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv30!, u0, (0.0, 10.0), p_mechanistic)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:net5_input1; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_input2 = PEtabParameter(:net5_input2; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_net5 = PEtabMLParameter(:net5; value = pnn)
pest = [p_alpha, p_beta, p_delta, p_input1, p_input2, p_net5]

conditions = PEtabCondition(:e1)

observables = [
    PEtabObservable(:prey_o, :prey, 0.05),
    PEtabObservable(:predator_o, :predator, 0.05)
]

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
