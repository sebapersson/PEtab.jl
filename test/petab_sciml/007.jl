test_case = "007"

nn_model = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
pnn, _st = Lux.setup(rng, nn_model)
const st = _st
nndict = Dict(:net1 => (st, nn_model))

function lv7!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv7!, u0, (0.0, 10.0), p_mechanistic)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_input1 = PEtabParameter(:input1; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = false)
p_input2 = PEtabParameter(:input2; scale = :lin, lb = 0.0, ub = 15.0, value = 1.0, estimate = true)
p_net1 = PEtabParameter(:p_net1; scale = :lin, lb = -15.0, ub = 15.0, value = 0.0)
pest = [p_alpha, p_beta, p_delta, p_input1, p_input2, p_net1]

mapping_table = Dict("net1" => [:input1 => :input1, :input2 => :input2, :output1 => :gamma])

conds = Dict("cond1" => Dict{Symbol, Symbol}())

obs_prey = PEtabObservable(:prey, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey" => obs_prey, "predator" => obs_predator)

path_m = joinpath(@__DIR__, "test_cases", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; nn = nndict,
                   mapping_table = mapping_table,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
for config in ProbConfigs
    petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = config.grad,
                                 split_over_conditions = config.split, sensealg=config.sensealg)
    test_model(test_case, petab_prob)
end
