using SciMLBase, Lux, ComponentArrays, PEtab, CSV, DataFrames, YAML,
      OrdinaryDiffEqRosenbrock, SciMLSensitivity, HDF5, Test
using Catalyst: @unpack
import Random
rng = Random.default_rng()

PROB_CONFIGS = [(grad = :ForwardDiff, split = false, sensealg = :ForwardDiff),
                (grad = :ForwardDiff, split = true, sensealg = :ForwardDiff),
                (grad = :ForwardEquations, split = false, sensealg = :ForwardDiff),
                (grad = :ForwardEquations, split = true, sensealg = :ForwardDiff),
                (grad = :ForwardEquations, split = true, sensealg = ForwardSensitivity()),
                (grad = :Adjoint, split = true, sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))]

function test_hybrid(test_case, petab_prob::PEtabODEProblem)
    @unpack split_over_conditions, gradient_method = petab_prob.probinfo
    @info "Case $(test_case) and gradient method $(gradient_method) and split = $(split_over_conditions)"
    # Reference values
    dirtest = joinpath(@__DIR__, "test_cases", "hybrid", test_case)
    path_solutions = joinpath(dirtest, "solutions.yaml")
    yamlfile = YAML.load_file(path_solutions)
    llh_ref, tol_llh = yamlfile["llh"], yamlfile["tol_llh"]
    tol_grad = yamlfile["tol_grad_llh"]
    gradfile_mech = yamlfile["grad_files"]["mech"]
    gradmech_ref = CSV.read(joinpath(dirtest, gradfile_mech), DataFrame)
    simfile, tol_sim = yamlfile["simulation_files"][1], yamlfile["tol_simulations"]
    simref = CSV.read(joinpath(dirtest, simfile), DataFrame)

    # Get Parameter values. If code is correct, network parameters should already be
    # set during import
    x = get_x(petab_prob)

    # PEtab problem values
    llh_petab = petab_prob.nllh(x) * -1
    grad_petab = petab_prob.grad(x) .* -1
    sim_petab = petab_prob.simulated_values(x)
    @test llh_petab ≈ llh_ref atol=tol_llh
    @test all(.≈(sim_petab, simref.simulation; atol=tol_sim))
    # Mechanistic parameters in gradient
    mechids = get_mechanistic_ids(petab_prob.model_info)
    for id in mechids
        iref = findfirst(x -> string(x) == "$id", gradmech_ref[!, :parameterId])
        @test grad_petab[id] ≈ gradmech_ref[iref, :value] atol=tol_grad
    end
    # Neural-net parameters
    for (ml_model_id, mm_model) in petab_prob.model_info.model.ml_models
        !(ml_model_id in petab_prob.model_info.xindices.xids[:ml_est]) && continue
        grad_test = grad_petab[ml_model_id]
        path_ref = joinpath(dirtest, yamlfile["grad_files"][string(ml_model_id)])
        grad_ref = deepcopy(grad_test)
        PEtab.set_ml_model_ps!(grad_ref, path_ref, mm_model.model, ml_model_id)
        @test all(.≈(grad_test, grad_ref; atol=tol_grad))
    end
    return nothing
end

function test_init(test_case, model::PEtabModel)::Nothing
    osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
    petab_prob = PEtabODEProblem(model; odesolver = osolver, gradient_method = :ForwardDiff)
    x = get_x(petab_prob)

    dirtest = joinpath(@__DIR__, "test_cases", "initialization", test_case)
    yamlfile = YAML.load_file(joinpath(dirtest, "solutions.yaml"))
    for (ml_model_id, ml_model) in petab_prob.model_info.model.ml_models
        ml_model_id = :net1
        !haskey(yamlfile["parameter_files"], string(ml_model_id)) && continue
        path_ref = joinpath(dirtest, yamlfile["parameter_files"][string(ml_model_id)])
        ps_ref = deepcopy(x[ml_model_id])
        PEtab.set_ml_model_ps!(ps_ref, path_ref, ml_model.model, ml_model_id)
        @test ps_ref == x[ml_model_id]
    end
    return nothing
end

function test_netimport(testcase, ml_model)::Nothing
    @info "Case $testcase"
    if testcase in ["003", "004", "005", "006", "007", "008", "009", "010", "014",
                    "015", "016", "017", "021", "022"]
        needs_batch = true
    else
        needs_batch = false
    end

    dirtest = joinpath(@__DIR__, "test_cases", "net_import", "$testcase")
    yaml_test = YAML.load_file(joinpath(dirtest, "solutions.yaml"))
    _ps, st = Lux.setup(rng, ml_model)
    ps = ComponentArray(_ps)

    # Expected input and output orders (in Julia and PyTorch for correct mapping)
    input_order_jl = yaml_test["input_order_jl"]
    input_order_py = yaml_test["input_order_py"]
    output_order_jl = yaml_test["output_order_jl"]
    output_order_py = yaml_test["output_order_py"]

    for j in 1:3
        # A subset of test-cases have multiple input arguments
        if testcase != "052"
            input_file = h5open(joinpath(dirtest, yaml_test["net_input"][j]), "r")
            _input = HDF5.read_dataset(input_file["inputs"]["input0"], "data")
            input = parse_array(_input, input_order_jl, input_order_py)
            close(input_file)
        else
            input_file1 = h5open(joinpath(dirtest, yaml_test["net_input_arg0"][j]), "r")
            input_file2 = h5open(joinpath(dirtest, yaml_test["net_input_arg1"][j]), "r")
            _input1 = HDF5.read_dataset(input_file1["inputs"]["input0"], "data")
            _input2 = HDF5.read_dataset(input_file2["inputs"]["input0"], "data")
            input = (_input1, _input2)
            close.([input_file1, input_file2])
        end
        # alpha dropout does not want mixed precision
        if testcase == "020"
            input = input |> f64
        end

        output_file = h5open(joinpath(dirtest, yaml_test["net_output"][j]), "r")
        _output = HDF5.read_dataset(output_file["outputs"]["output0"], "data")
        output_ref = parse_array(_output, output_order_jl, output_order_py)
        close(output_file)

        if needs_batch
            input = reshape(input, (size(input)..., 1))
            output_ref = reshape(output_ref, (size(output_ref)..., 1))
        end

        if haskey(yaml_test, "net_ps")
            path_h5 = joinpath(dirtest, yaml_test["net_ps"][j])
            PEtab.set_ml_model_ps!(ps, path_h5, ml_model, :net0)
        end

        if haskey(yaml_test, "dropout")
            testtol = 2e-2
            output = zeros(size(output_ref))
            nsamples = yaml_test["dropout"]
            for i in 1:nsamples
                _output, st = ml_model(input, ps, st)
                output .+= _output
            end
            output ./= nsamples
        else
            testtol = 1e-3
            output, st = ml_model(input, ps, st)
        end
        @test all(.≈(output, output_ref; atol = testtol))
    end
    return nothing
end

function get_mechanistic_ids(model_info::PEtab.ModelInfo)::Vector{Symbol}
    mechanistic_ids = Symbol[]
    for id in model_info.xindices.xids[:estimate]
        id in model_info.xindices.xids[:ml] && continue
        push!(mechanistic_ids, id)
    end
    return mechanistic_ids
end

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
