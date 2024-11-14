using YAML, PEtab, Lux, Random, ComponentArrays, CSV, DataFrames, Test
rng = Random.default_rng()

function df_to_array(df::DataFrame, order_jl::Vector{String}, order_py::Vector{String})
    if df[!, :ix] isa Vector{Int64}
        ix = df[!, :ix] .+ 1
        dims = (length(ix), )
    else
        ix = [Tuple(parse.(Int64, split(ix, ";")) .+ 1) for ix in df[!, :ix]]
        dims = maximum(ix)
    end
    out = zeros(dims)
    for i in eachindex(ix)
        out[ix[i]...] = df[i, :value]
    end
    length(size(out)) == 1 && return out
    # At this point the array follows a multidimensional PyTorch indexing. Therefore the
    # array must be reshaped to Julia indexing
    imap = zeros(Int64, length(order_jl))
    for i in eachindex(order_jl)
        imap[i] = findfirst(x -> x == order_jl[i], order_py)
    end
    map = collect(1:length(order_py)) .=> imap
    return PEtab._reshape_array(out, map)
end

@testset "Neural network import" begin
    for i in 1:49
        testcase = i < 10 ? "00$i" : "0$i"
        @info "Case $testcase"
        if testcase in ["003", "004", "005", "006", "007", "008", "009", "010", "014",
                        "015", "016", "017", "021", "022"]
            needs_batch = true
        else
            needs_batch = false
        end

        dirtest = joinpath(@__DIR__, "test_cases", "net_$testcase")
        yaml_test = YAML.load_file(joinpath(dirtest, "solutions.yaml"))
        nnmodel = PEtab.parse_to_lux(joinpath(dirtest, yaml_test["net_file"]))
        _ps, st = Lux.setup(rng, nnmodel)
        ps = ComponentArray(_ps)

        # Expected input and output orders (in Julia and PyTorch for correct mapping)
        input_order_jl = yaml_test["input_order_jl"]
        input_order_py = yaml_test["input_order_py"]
        output_order_jl = yaml_test["output_order_jl"]
        output_order_py = yaml_test["output_order_py"]

        for j in 1:3
            input_df = CSV.read(joinpath(dirtest, yaml_test["net_input"][j]), DataFrame)
            # alpha dropout does not want mixed precision
            if i == 20
                input = df_to_array(input_df, input_order_jl, input_order_py) |> f64
            else
                input = df_to_array(input_df, input_order_jl, input_order_py) |> f32
            end
            output_df = CSV.read(joinpath(dirtest, yaml_test["net_output"][j]), DataFrame)
            output_ref = df_to_array(output_df, output_order_jl, output_order_py)
            if needs_batch
                input = reshape(input, (size(input)..., 1))
                output_ref = reshape(output_ref, (size(output_ref)..., 1))
            end

            if haskey(yaml_test, "net_ps")
                df_ps = CSV.read(joinpath(dirtest, yaml_test["net_ps"][j]), DataFrame)
                PEtab.set_ps_net!(ps, df_ps, :net, nnmodel)
            end

            if haskey(yaml_test, "dropout")
                testtol = 1e-2
                output = zeros(size(output_ref))
                nsamples = yaml_test["dropout"]
                for i in 1:nsamples
                    _output, st = nnmodel(input, ps, st)
                    output .+= _output
                end
                output ./= nsamples
            else
                testtol = 1e-3
                output, st = nnmodel(input, ps, st)
            end
            @test all(.≈(output, output_ref; atol = testtol))
        end
    end
end

# TODO: Fix the world-problem with the net

testcase = "002"


foo = [0.49123794956695477 0.3597643032282879; 0.7362397216065666 0.6042561612489359; 0.2180631632767165 0.6576552270213534; 0.7664790825976505 0.8393876937604513; 0.265279355014322 0.3040053888556315; 0.35466378364637213 0.32520908629540424; 0.5101723145785534 0.4721397687852346; -0.021263638759468892 0.3336367154612331; 0.4067012993386193 0.0023671212869858516; 0.34435485916993064 0.5162486093204173;;;;]


path_yaml = joinpath(@__DIR__, "nets/activation_f.yaml")
nn_model = load_lux_net(path_yaml)
rng = Random.default_rng()
pnn, st = Lux.setup(rng, nn_model)
input = ones(2)
output = nn_model(input, pnn, st)[1]

input = ones(100, 100, 100, 1, 1)
model = FlattenLayer(; N = 2)
model = Chain(MaxPool((5, 4, 3)), FlattenLayer(; N = 2))
input = ones(100, 100, 100, 1, 1)
model = Dense(5, 5)
model = Chain(Recurrence(RNNCell(5 => 100); return_sequence=true),
              Recurrence(RNNCell(100 => 100); return_sequence=true),
              Recurrence(RNNCell(100 => 10); return_sequence=false))


model = LayerNorm((5, 10); dims = 2)
ps, st = Lux.setup(rng, model)
input = ones(5, 10, 20)
y = model(input, ps, st)[1]





using Catalyst: @unpack
using DataFrames, ComponentArrays, Lux, ComponentArrays, Random
rng = Random.default_rng()
layer = Lux.Conv((10, ), 2 => 1)

layer = Lux.Dense(5 => 20)
ps, st = Lux.setup(rng, layer)

ps_new = (ComponentArray(ps) |> deepcopy) .* 0.0
df_ps = layer_ps_to_tidy(layer, ps, :foo, :tada)
set_ps_layer!(ps_new, layer, df_ps)::Nothing
