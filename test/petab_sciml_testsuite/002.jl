test_case = "002"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

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

# Test with constant Vector input
ml_models = MLModel(:net1, nn2, true; inputs = [1.0, 1.0], outputs = [:gamma])
path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn = Lux.initialparameters(rng, nn2) |> ComponentArray |> f64
PEtab._set_ml_model_ps!(pnn, path_h5, nn2, :net1)

function lv2!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv2!, u0, (0.0, 10.0), p_mechanistic)

pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
    PEtabMLParameter(:net1; value = pnn),
]

observables = [
    PEtabObservable(:prey_o, :prey, 0.05),
    PEtabObservable(:predator_o, :predator, 0.05),
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
