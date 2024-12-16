using YAML, PEtab, Lux, Random, ComponentArrays, HDF5, Test
rng = Random.default_rng()

function parse_array(x::Array{T}, order_jl::Vector{String}, order_py::Vector{String})::Array{T} where T <: AbstractFloat
    # To column-major
    out = permutedims(x, reverse(1:ndims(x)))
    length(size(out)) == 1 && return out
    # At this point the array follows a multixdimensional PyTorch indexing. Therefore the
    # array must be reshaped to Julia indexing
    imap = zeros(Int64, length(order_jl))
    for i in eachindex(order_jl)
        imap[i] = findfirst(x -> x == order_jl[i], order_py)
    end
    map = collect(1:length(order_py)) .=> imap
    return PEtab._reshape_array(out, map)
end

@testset "Neural network import" begin
    for i in 1:51
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
        nnmodel, netid = PEtab.parse_to_lux(joinpath(dirtest, yaml_test["net_file"]))
        _ps, st = Lux.setup(rng, nnmodel)
        ps = ComponentArray(_ps)

        # Expected input and output orders (in Julia and PyTorch for correct mapping)
        input_order_jl = yaml_test["input_order_jl"]
        input_order_py = yaml_test["input_order_py"]
        output_order_jl = yaml_test["output_order_jl"]
        output_order_py = yaml_test["output_order_py"]

        for j in 1:3
            _input = h5read(joinpath(dirtest, yaml_test["net_input"][j]), "input")
            input = parse_array(_input, input_order_jl, input_order_py)
            # alpha dropout does not want mixed precision
            if i == 20
                input = input |> f64
            end
            _output = h5read(joinpath(dirtest, yaml_test["net_output"][j]), "output")
            output_ref = parse_array(_output, output_order_jl, output_order_py)
            if needs_batch
                input = reshape(input, (size(input)..., 1))
                output_ref = reshape(output_ref, (size(output_ref)..., 1))
            end

            if haskey(yaml_test, "net_ps")
                path_h5 = joinpath(dirtest, yaml_test["net_ps"][j])
                PEtab.set_ps_net!(ps, path_h5, nnmodel)
            end

            if haskey(yaml_test, "dropout")
                testtol = 2e-2
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
