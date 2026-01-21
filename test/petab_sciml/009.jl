test_case = "009"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn9_1 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nn9_2 = @compact(
    layer1 = Dense(2, 5, Lux.relu),
    layer2 = Dense(5, 10, Lux.relu),
    layer3 = Dense(10, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_models = Dict(:net1 => MLModel(nn9_1; static = false, inputs = [:prey, :predator], outputs = [:net1_output1]),
                 :net2 => MLModel(nn9_2; static = false, inputs = [:alpha, :predator], outputs = [:net2_output1]))
path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn1 = Lux.initialparameters(rng, nn9_1) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn1, path_h5, nn9_1, :net1)
path_h5 = joinpath(dir_case, "net2_ps.hdf5")
pnn2 = Lux.initialparameters(rng, nn9_2) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn2, path_h5, nn9_2, :net2)

function lv9!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv9!, u0, (0.0, 10.0), p_mechanistic)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_gamma = PEtabParameter(:gamma; scale = :lin, lb = 0.0, ub = 15.0, value = 0.8)
p_net1 = PEtabMLParameter(:net1, true, pnn1)
p_net2 = PEtabMLParameter(:net2, true, pnn2)
pest = [p_alpha, p_beta, p_delta, p_gamma, p_net1, p_net2]

conditions = PEtabCondition(:e1)

observables = [
    PEtabObservable(:prey_o, :net1_output1, 0.05),
    PEtabObservable(:predator_o, :net2_output1, 0.05)
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
