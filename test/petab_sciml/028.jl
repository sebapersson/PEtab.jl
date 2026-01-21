test_case = "028"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn28 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nn28_frozen = @compact(
    layer1 = Lux.Experimental.freeze(Dense(2, 5, Lux.tanh), (:weight, )),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
# Setup parameters for network to use during estimation
path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn = Lux.initialparameters(rng, nn28_frozen) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn28_frozen, :net1)
# Set frozen parameters
pnn_tmp = Lux.initialparameters(rng, nn28) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn_tmp, path_h5, nn28, :net1)
st = Lux.initialstates(rng, nn28_frozen) |> f64
st.layer1.frozen_params.weight .= pnn_tmp.layer1.weight
# Given this ml_model can be built
ml_models = Dict(:net1 => MLModel(nn28_frozen; st = st, static = false, inputs = [:prey, :predator], outputs = [:net1_output1]))

function lv28!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv28!, u0, (0.0, 10.0), p_mechanistic)

# Setup the PEtabModel as usual
# (algebraic expressions for initial values, specie-map overrides, will have to test later)
p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_gamma = PEtabParameter(:gamma; scale = :lin, lb = 0.0, ub = 15.0, value = 0.8)
p_net1 = PEtabMLParameter(:net1, true, pnn)
pest = [p_alpha, p_beta, p_delta, p_gamma, p_net1]

observables = [
    PEtabObservable(:prey_o, :net1_output1, 0.05),
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
