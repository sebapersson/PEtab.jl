test_case = "001"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

# Define model as ODEProblem
# runic: off
nn1 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
# runic: on

ml_model = MLModel(:net1, nn1, false)
path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn = Lux.initialparameters(rng, nn1) |> ComponentArray |> f64
PEtab._set_ml_model_ps!(pnn, path_h5, nn1, :net1)

function lv1!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p

    net1 = ml_models[:net1]
    du_nn, st = net1.lux_model([prey, predator], p[:net1], net1.st)
    net1.st = st

    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = du_nn[1] - delta * predator # predator
    return nothing
end

p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9)
u0 = (prey = 0.44249296, predator = 4.6280594)
uprob = UDEProblem(lv1!, u0, (0.0, 10.0), p_mechanistic, ml_model)

# Define model as ODESystem
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

# Test as ODEProblem
model_prob = PEtabModel(
    uprob, observables, measurements, pest; ml_models = ml_model,
    simulation_conditions = conditions
)
petab_prob_prob = PEtabODEProblem(
    model_prob; odesolver = ode_solver, gradient_method = :ForwardDiff
)
test_hybrid(test_case, petab_prob_prob)

# Test as ODESystem. As PEtab-SciML import uses ODEProblem, must test all gradient methods
model_sys = PEtabModel(
    sys_ude, observables, measurements, pest; simulation_conditions = conditions
)
for config in PROB_CONFIGS
    petab_prob_sys = PEtabODEProblem(
        model_sys; odesolver = ode_solver, gradient_method = config.grad,
        split_over_conditions = config.split, sensealg = config.sensealg
    )
    test_hybrid(test_case, petab_prob_sys)
end
# BacksolveAdjoint must be tested on non-stiff model
petab_prob_sys = PEtabODEProblem(
    model_sys; odesolver = ode_solver, gradient_method = :Adjoint,
    sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))
)
test_hybrid(test_case, petab_prob_sys)

model_rn = PEtabModel(
    rn_ude, observables, measurements, pest; simulation_conditions = conditions
)
petab_prob_rn = PEtabODEProblem(model_rn; odesolver = ode_solver)
test_hybrid(test_case, petab_prob_rn)

# Check that parameters to estimated throw correctly on bad input
pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
]
@test_throws PEtab.PEtabInputError begin
    model = PEtabModel(
        uprob, observables, measurements, pest; ml_models = ml_model,
        simulation_conditions = conditions
    )
end
@test_throws PEtab.PEtabInputError begin
    model = PEtabModel(
        sys_ude, observables, measurements, pest; simulation_conditions = conditions
    )
end

pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
    3.0,
]
@test_throws PEtab.PEtabInputError begin
    model = PEtabModel(
        uprob, observables, measurements, pest; ml_models = ml_model,
        simulation_conditions = conditions
    )
end

# ODESystem where parameter maps incorrectly to two NN-models
nn1_chain = Lux.Chain(
    layer1 = Dense(2 => 5, Lux.tanh),
    layer2 = Dense(5 => 5, Lux.tanh),
    layer3 = Dense(5 => 1)
)
@SymbolicNeuralNetwork NN1, net1 = nn1_chain
@SymbolicNeuralNetwork NN2, _ = nn1_chain
@variables prey(t) = 0.44249296 predator(t) = 4.6280594
@parameters alpha beta delta
eqs_ude = [
    D(prey) ~ alpha * prey - beta * prey * predator
    D(predator) ~ NN1([prey, predator], net1)[1] + NN2([prey, predator], net1)[1] - delta * predator
]
@mtkcompile sys_ude = System(eqs_ude, t)

pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
    PEtabMLParameter(:net1; value = pnn),
]
@test_throws PEtab.PEtabInputError begin
    model = PEtabModel(
        sys_ude, observables, measurements, pest; simulation_conditions = conditions
    )
end

# Parameter not associated with ML-model
nn1_chain = Lux.Chain(
    layer1 = Dense(2 => 5, Lux.tanh),
    layer2 = Dense(5 => 5, Lux.tanh),
    layer3 = Dense(5 => 1)
)
@SymbolicNeuralNetwork NN1, net1 = nn1_chain
@variables prey(t) = 0.44249296 predator(t) = 4.6280594
@parameters alpha beta delta
eqs_ude = [
    D(prey) ~ alpha * prey - beta * prey * predator
    D(predator) ~ net1[1] - delta * predator
]
@mtkcompile sys_ude = System(eqs_ude, t)
@test_throws PEtab.PEtabInputError begin
    model = PEtabModel(
        sys_ude, observables, measurements, pest; simulation_conditions = conditions
    )
end
