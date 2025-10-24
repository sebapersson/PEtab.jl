test_case = "029"

nn5 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do (x1, x2)
    x = cat(x1, x2; dims = 1)
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
ml_models = Dict(:net5 => MLModel(nn5; static = false))
path_h5 = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "net5_ps.hdf5")
pnn = Lux.initialparameters(rng, nn5) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn5, :net5)

function _lv29!(du, u, p, t, ml_models)
    prey, predator = u
    @unpack alpha, delta, beta = p
    net5 = ml_models[:net5]
    du_nn, st = net5.model(([prey], [predator]), p[:net5], net5.st)
    net5.st = st

    du[1] = alpha*prey - beta * prey * predator # prey
    du[2] = du_nn[1] - delta*predator # predator
    return nothing
end
lv29! = let _ml_models = ml_models
    (du, u, p, t) -> _lv29!(du, u, p, t, _ml_models)
end

p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
p_ode = ComponentArray(merge(p_mechanistic, (net5=pnn,)))
u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
uprob = ODEProblem(lv29!, u0, (0.0, 10.0), p_ode)

p_alpha = PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3)
p_beta = PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9)
p_delta = PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8)
p_net5 = PEtabMLParameter(:net5, true, pnn)
pest = [p_alpha, p_beta, p_delta, p_net5]

obs_prey = PEtabObservable(:prey, 0.05)
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
