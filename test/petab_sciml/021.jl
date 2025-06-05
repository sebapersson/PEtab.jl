test_case = "021"

nn21 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 2)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
nnmodels = Dict(:net4 => NNModel(nn21; static = true, inputs = [:net1_input_pre1, :net1_input_pre2], outputs = [:alpha, :gamma]))
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net4_ps.hdf5")
pnn = Lux.initialparameters(rng, nn21) |> ComponentArray |> f64
PEtab.set_ps_net!(pnn, path_h5, nn21)

function lv21!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv21!, u0, (0.0, 10.0), p_mechanistic)

p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:net1_input_pre1; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_input2 = PEtabParameter(:net1_input_pre2; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_net4 = PEtabNetParameter(:net4, true, pnn)
pest = [p_beta, p_delta, p_input1, p_input2, p_net4]

obs_prey = PEtabObservable(:prey, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey_o" => obs_prey, "predator_o" => obs_predator)

conds = Dict("cond1" => Dict{Symbol, Symbol}())

path_m = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; nnmodels = nnmodels,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = :ForwardDiff,
                             split_over_conditions = true)
test_hybrid(test_case, petab_prob)
