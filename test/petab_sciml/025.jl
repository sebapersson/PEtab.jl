test_case = "025"

nn25 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nn25_frozen = @compact(
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
pnn = Lux.initialparameters(rng, nn25_frozen) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn25_frozen, :net1)
# Set frozen parameters
pnn_tmp = Lux.initialparameters(rng, nn25) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn_tmp, path_h5, nn25, :net1)
st = Lux.initialstates(rng, nn25_frozen) |> f64
st.layer1.frozen_params.weight .= pnn_tmp.layer1.weight
st.layer1.frozen_params.bias .= pnn_tmp.layer1.bias
# Given this ml_model can be built
ml_models = Dict(:net1 => MLModel(nn25_frozen; st = st, static = false, inputs = [:prey, :predator], outputs = [:net1_output1]))

function lv25!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv25!, u0, (0.0, 10.0), p_mechanistic)

# Setup the PEtabModel as usual
# (algebraic expressions for initial values, specie-map overrides, will have to test later)
p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_gamma = PEtabParameter(:gamma; scale = :lin, lb = 0.0, ub = 15.0, value = 0.8)
p_net1 = PEtabMLParameter(:net1, true, pnn)
pest = [p_alpha, p_beta, p_delta, p_gamma, p_net1]

obs_prey = PEtabObservable(:net1_output1, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey_o" => obs_prey, "predator_o" => obs_predator)

conds = Dict("cond1" => Dict{Symbol, Symbol}())

path_m = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; ml_models = ml_models,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = :ForwardDiff,
                             split_over_conditions = true)
test_hybrid(test_case, petab_prob)
