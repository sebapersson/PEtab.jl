test_case = "011"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

# runic: off
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
# runic: on

# Model as ODEProblem
ml_model1 = MLModel(:net1, nn11_1, false)
ml_model2 = MLModel(:net2, nn11_2, true; inputs = [:net2_input_pre1, :net2_input_pre1], outputs = [:beta])
ml_models = MLModels(ml_model1, ml_model2)

path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn1 = Lux.initialparameters(rng, nn11_1) |> ComponentArray |> f64
PEtab._set_ml_model_ps!(pnn1, path_h5, nn11_1, :net1)
path_h5 = joinpath(dir_case, "net2_ps.hdf5")
pnn2 = Lux.initialparameters(rng, nn11_2) |> ComponentArray |> f64
PEtab._set_ml_model_ps!(pnn2, path_h5, nn11_2, :net2)

function lv11!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p

    net1 = ml_models[:net1]
    du1_nn, st = net1.lux_model([prey, predator], p[:net1], net1.st)
    net1.st = st

    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = du1_nn[1] - delta * predator # predator
    return nothing
end

p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = UDEProblem(lv11!, u0, (0.0, 10.0), p_mechanistic, ml_model1)

# Model as ODESystem
nn1_chain = Lux.Chain(
    layer1 = Dense(2 => 5, Lux.tanh),
    layer2 = Dense(5 => 5, Lux.tanh),
    layer3 = Dense(5 => 1)
)
@SymbolicNeuralNetwork NN, net1 = nn1_chain
@variables prey(t) = 0.44249296 predator(t) = 4.6280594
@parameters alpha beta delta
eqs_ude = [
    D(prey) ~ alpha * prey - beta * prey * predator
    D(predator) ~ NN([prey, predator], net1)[1] - delta * predator
]
@mtkcompile sys_ude = System(eqs_ude, t)

# Define the model as a Catalyst ReactionSystem
NN_rate(x, y) = NN([x, y], net1)[1]
rn_ude = @reaction_network begin
    @species begin
        prey(t) = 0.44249296
        predator(t) = 4.6280594
    end
    @parameters begin
        alpha
        beta
        delta
    end
    alpha, prey --> 2prey
    beta, prey + predator --> predator
    $NN_rate(prey, predator), 0 --> predator
    delta, predator --> 0
end

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:net2_input_pre1; scale = :lin, lb = 0.0, ub = 15.0, value = 2.0, estimate = false)
p_input2 = PEtabParameter(:net2_input_pre2; scale = :lin, lb = 0.0, ub = 15.0, value = 2.0, estimate = false)
p_net1 = PEtabMLParameter(:net1; value = pnn1)
p_net2 = PEtabMLParameter(:net2; value = pnn2)
pest = [p_alpha, p_delta, p_input1, p_input2, p_net1, p_net2]

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

# Test as ODESystem. As PEtab-SciML import uses ODEProblem, must test all gradient methods
ml_model = ml_models[:net2]
model_sys = PEtabModel(
    sys_ude, observables, measurements, pest, simulation_conditions = conditions,
    ml_models = ml_model
)
for config in PROB_CONFIGS
    petab_prob_sys = PEtabODEProblem(
        model_sys; odesolver = ode_solver, gradient_method = config.grad,
        split_over_conditions = config.split, sensealg = config.sensealg
    )
    test_hybrid(test_case, petab_prob_sys)
end

model_rn = PEtabModel(
    rn_ude, observables, measurements, pest; simulation_conditions = conditions,
    ml_models = ml_model
)
petab_prob_rn = PEtabODEProblem(model_rn; odesolver = ode_solver)
test_hybrid(test_case, petab_prob_rn)
