using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

test_case = "003"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn3 = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end

ml_models = MLModel(
    :net1, nn3, true; inputs = inputs = [:net1_input1, :net1_input2], outputs = [:gamma]
)
path_h5 = joinpath(dir_case, "net1_ps.hdf5")
pnn = Lux.initialparameters(rng, nn3) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn3, :net1)

@mtkmodel _SYS3 begin
    @parameters begin
        alpha
        delta
        beta
        gamma
    end
    @variables begin
        prey(t) = 0.44249296
        predator(t) = 4.6280594
    end
    @equations begin
        D(prey) ~ alpha * prey - beta * prey * predator
        D(predator) ~ gamma * predator * prey - delta * predator
    end
end
@mtkbuild sys = _SYS3()

pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
    PEtabParameter(:net1_input_pre1; scale = :lin, value = 1.0, estimate = false),
    PEtabParameter(:net1_input_pre2; scale = :lin, value = 1.0, estimate = false),
    PEtabMLParameter(:net1; value = pnn)
]

conditions = [
    PEtabCondition(:e1, :net1_input1 => 10.0, :net1_input2 => 20.0),
    PEtabCondition(:e2, :net1_input1 => :net1_input_pre1, :net1_input2 => :net1_input_pre2)
]

observables = [
    PEtabObservable(:prey_o, :prey, 0.05),
    PEtabObservable(:predator_o, :predator, 0.05)
]

path_m = joinpath(dir_case, "measurements.tsv")
measurements = CSV.read(path_m, DataFrame)
rename!(measurements, "experimentId" => "simulation_id")

model = PEtabModel(
    sys, observables, measurements, pest;
    ml_models = ml_models, simulation_conditions = conditions
)
petab_prob = PEtabODEProblem(
    model; odesolver = ode_solver, gradient_method = :ForwardDiff,
    split_over_conditions = true
)
test_hybrid(test_case, petab_prob)
