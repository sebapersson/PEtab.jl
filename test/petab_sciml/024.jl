test_case = "024"

nn24 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nn24_frozen = @compact(
    layer1 = Lux.Experimental.freeze(Dense(2, 5, Lux.tanh), (:weight, :bias)),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
# Setup parameters for network to use during estimation
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net1_ps.hdf5")
pnn = Lux.initialparameters(rng, nn24_frozen) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn24_frozen)
# Set frozen parameters
pnn_tmp = Lux.initialparameters(rng, nn24) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn_tmp, path_h5, nn24)
st = Lux.initialstates(rng, nn24_frozen) |> f64
st.layer1.frozen_params.weight .= pnn_tmp.layer1.weight
st.layer1.frozen_params.bias .= pnn_tmp.layer1.bias
# Given this ml_model can be built
ml_models = Dict(:net1 => MLModel(nn24_frozen; st = st, static = true, inputs = [:net1_input1, :net1_input2], outputs = [:gamma]))

function lv24!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv24!, u0, (0.0, 10.0), p_mechanistic)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:net1_input1; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_input2 = PEtabParameter(:net1_input2; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_net1 = PEtabMLParameter(:net1, true, pnn)
pest = [p_alpha, p_beta, p_delta, p_input1, p_input2, p_net1]

conds = Dict("cond1" => Dict{Symbol, Symbol}())

obs_prey = PEtabObservable(:prey, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey_o" => obs_prey, "predator_o" => obs_predator)

path_m = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; ml_models = ml_models,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = :ForwardDiff,
                             split_over_conditions = true)
test_hybrid(test_case, petab_prob)
