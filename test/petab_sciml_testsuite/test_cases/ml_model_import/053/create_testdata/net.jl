using Lux, StableRNGs
using PEtabSciMLTestsuite: save_ps, save_io, write_yaml

nn_model = @compact(
    layer1 = Dense(20, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do (x1, x2)
    embed = cat(x1, x2; dims = 1)
    embed = layer1(embed)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end

input_order_jl, input_order_py = ["W"], ["W"]
output_order_jl, output_order_py = ["W"], ["W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input1 = rand(rng, Float32, 10)
    input2 = rand(rng, Float32, 10)
    output = nn_model((input1, input2), ps, st)[1]
    save_ps(dirsave, i, nn_model, :net0, ps)
    save_io(dirsave, i, input1, input_order_jl, input_order_py, :input; arg_index = 0)
    save_io(dirsave, i, input2, input_order_jl, input_order_py, :input; arg_index = 1)
    save_io(dirsave, i, output[1, :], output_order_jl, output_order_py, :output)
end
write_yaml(
    dirsave, input_order_jl, input_order_py,
    output_order_jl, output_order_py; n_input_args = 2
)
