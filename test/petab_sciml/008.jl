test_case = "008"

nn8_1 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nn8_2 = @compact(
    layer1 = Dense(2, 5, Lux.relu),
    layer2 = Dense(5, 10, Lux.relu),
    layer3 = Dense(10, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nnmodels = Dict(:net1 => NNModel(nn8_1, inputs = [:net1_input_pre1, :net1_input_pre2], outputs = [:gamma]),
                :net2 => NNModel(nn8_2, inputs = [:net2_input_pre1, :net2_input_pre2], outputs = [:beta]))
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net1_ps.hdf5")
pnn1 = Lux.initialparameters(rng, nn8_1) |> ComponentArray |> f64
PEtab.set_ps_net!(pnn1, path_h5, nn8_1)
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net2_ps.hdf5")
pnn2 = Lux.initialparameters(rng, nn8_2) |> ComponentArray |> f64
PEtab.set_ps_net!(pnn2, path_h5, nn8_2)

function lv8!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv8!, u0, (0.0, 10.0), p_mechanistic)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:net1_input_pre1; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_input2 = PEtabParameter(:net1_input_pre2; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_input3 = PEtabParameter(:net2_input_pre1; scale = :lin, lb = 0.0, ub = 15.0, value = 2.0, estimate = false)
p_input4 = PEtabParameter(:net2_input_pre2; scale = :lin, lb = 0.0, ub = 15.0, value = 2.0, estimate = false)
p_net1 = PEtabNetParameter(:net1, true, pnn1)
p_net2 = PEtabNetParameter(:net2, true, pnn2)
pest = [p_alpha, p_delta, p_input1, p_input2, p_input3, p_input4, p_net1, p_net2]

conds = Dict("cond1" => Dict{Symbol, Symbol}())

obs_prey = PEtabObservable(:prey, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey_o" => obs_prey, "predator_o" => obs_predator)

path_m = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; nnmodels = nnmodels,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = :ForwardDiff,
                             split_over_conditions = true)
test_hybrid(test_case, petab_prob)
