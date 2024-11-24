test_case = "005"

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
nndict = Dict(:net1 => [st, nn_model])

function lv5!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv5!, u0, (0.0, 10.0), p_mechanistic)

# Setup the PEtabModel as usual
# (algebraic expressions for initial values, specie-map overrides, will have to test later)
p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_gamma = PEtabParameter(:gamma; scale = :lin, lb = 0.0, ub = 15.0, value = 0.8)
p_net1 = PEtabParameter(:p_net1; scale = :lin, lb = -15.0, ub = 15.0, value = 0.0)
pest = [p_alpha, p_beta, p_delta, p_gamma, p_net1]

mapping_table = Dict("net1" => [:input1 => :prey, :input2 => :predator, :output1 => :net1_output1])

obs_prey = PEtabObservable(:net1_output1, 0.05)
obs_predator = PEtabObservable(:predator, 0.05)
obs = Dict("prey" => obs_prey, "predator" => obs_predator)

conds = Dict("cond1" => Dict{Symbol, Symbol}())

path_m = joinpath(@__DIR__, "test_cases", test_case, "petab", "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)

model = PEtabModel(uprob, obs, measurements, pest; nn = nndict,
                   mapping_table = mapping_table,
                   simulation_conditions = conds)
osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
for config in PROB_CONFIGS
    petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = config.grad,
                                 split_over_conditions = config.split, sensealg=config.sensealg)
    test_model(test_case, petab_prob)
end
