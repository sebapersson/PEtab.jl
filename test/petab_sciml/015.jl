test_case = "015"
dir_case = joinpath(@__DIR__, "test_cases", "sciml_problem_import", test_case, "petab")

nn15 = @compact(
    layer1=Conv((5, 5), 3=>1; cross_correlation = true),
    layer2=FlattenLayer(),
    layer3=Dense(36=>1, Lux.relu)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end

# Input data, ensure correct ordering for Julia
input_hdf5 = HDF5.h5open(joinpath(dir_case, "net3_input2.hdf5"), "r")
input_data1 = HDF5.read_dataset(input_hdf5["inputs"]["input0"], "cond1")
input_data1 = permutedims(input_data1, reverse(1:ndims(input_data1)))
input_data1 = PEtab._reshape_io_data(input_data1)
input_data1 = reshape(input_data1, (size(input_data1)..., 1)) |> f64
input_data2 = HDF5.read_dataset(input_hdf5["inputs"]["input0"], "cond2")
input_data2 = permutedims(input_data2, reverse(1:ndims(input_data2)))
input_data2 = PEtab._reshape_io_data(input_data2)
input_data2 = reshape(input_data2, (size(input_data2)..., 1)) |> f64
close(input_hdf5)

ml_models = Dict(:net3 => MLModel(nn15; static = true, inputs = [:net3_input], outputs = [:gamma]))
path_h5 = joinpath(dir_case, "net3_ps.hdf5")
pnn = Lux.initialparameters(rng, nn15) |> ComponentArray |> f64
PEtab.set_ml_model_ps!(pnn, path_h5, nn15, :net3)

# Array input needs to be carried into xindices, in which it can be used to build the maps.
# How? They need to be stored somewhere I can transfer them to ParameterIndices, but
# they cannot be table stored..., will store in MLModel (it will always be available,
# and can via this approach avoid any copy of it!!

function lv15!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * predator * prey - delta * predator # predator
    return nothing
end

u0 = ComponentArray(prey = 0.44249296, predator = 4.6280594)
p_mechanistic = ComponentArray(alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
uprob = ODEProblem(lv15!, u0, (0.0, 10.0), p_mechanistic)

pest = [
    PEtabParameter(:alpha; scale = :lin, lb = 0.0, ub = 15.0, value = 1.3),
    PEtabParameter(:beta; scale = :lin, lb = 0.0, ub = 15.0, value = 0.9),
    PEtabParameter(:delta; scale = :lin, lb = 0.0, ub = 15.0, value = 1.8),
    PEtabMLParameter(:net3, true, pnn)
]

observables = [
    PEtabObservable(:prey_o, :prey, 0.05),
    PEtabObservable(:predator_o, :predator, 0.05)
]

conditions = [
    PEtabCondition(:e1, :net3_input => input_data1),
    PEtabCondition(:e2, :net3_input => input_data2)
]

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
