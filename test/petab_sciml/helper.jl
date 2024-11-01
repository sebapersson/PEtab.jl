using SciMLBase, Lux, ComponentArrays, PEtab, CSV, DataFrames, YAML,
      OrdinaryDiffEqRosenbrock, SciMLSensitivity, Test
using Catalyst: @unpack
import Random
rng = Random.default_rng()

ProbConfigs = [(grad = :ForwardDiff, split = false, sensealg = :ForwardDiff),
               (grad = :ForwardDiff, split = true, sensealg = :ForwardDiff),
               (grad = :ForwardEquations, split = false, sensealg = :ForwardDiff),
               (grad = :ForwardEquations, split = true, sensealg = :ForwardDiff),
               (grad = :ForwardEquations, split = true, sensealg = ForwardSensitivity()),
               (grad = :Adjoint, split = true, sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))]

function get_mechanistic_ids(model_info::PEtab.ModelInfo)::Vector{Symbol}
    mechanistic_ids = Symbol[]
    for id in model_info.xindices.xids[:estimate]
        id in model_info.xindices.xids[:nn] && continue
        push!(mechanistic_ids, id)
    end
    return mechanistic_ids
end

function get_x_nn(test_case::String, petab_prob::PEtabODEProblem)
    path_ps_net = joinpath(@__DIR__, "test_cases", test_case, "petab", "parameters_nn.tsv")
    df_ps_net = CSV.read(path_ps_net, DataFrame)
    x = get_x(petab_prob)
    for netid in keys(model.nn)
        pid = Symbol("p_$(netid)")
        PEtab.set_ps_net!((@view x[pid]), df_ps_net, netid, model.nn[netid][2])
    end
    return x
end

function test_model(test_case, petab_prob::PEtabODEProblem)
    @unpack split_over_conditions, gradient_method = petab_prob.probinfo
    @info "Case $(test_case) and gradient method $(gradient_method) and split = $(split_over_conditions)"
    # Reference values
    test_dir = joinpath(@__DIR__, "test_cases", test_case)
    path_solutions = joinpath(test_dir, "solutions.yaml")
    yamlfile = YAML.load_file(path_solutions)
    llh_ref, tol_llh = yamlfile["llh"], yamlfile["tol_llh"]
    gradfile, tol_grad = yamlfile["grad_llh_files"][1], yamlfile["tol_grad_llh"]
    grad_ref = CSV.read(joinpath(test_dir, gradfile), DataFrame)
    simfile, tol_sim = yamlfile["simulation_files"][1], yamlfile["tol_simulations"]
    simref = CSV.read(joinpath(test_dir, simfile), DataFrame)

    # PEtab problem values
    x = get_x_nn(test_case, petab_prob)
    llh_petab = petab_prob.nllh(x) * -1
    grad_petab = petab_prob.grad(x) .* -1
    sim_petab = petab_prob.simulated_values(x)

    @test llh_petab ≈ llh_ref atol=tol_llh
    @test all(.≈(sim_petab, simref.simulation; atol=tol_sim))
    # Mechanistic parameters in gradient
    mechids = get_mechanistic_ids(petab_prob.model_info)
    for id in mechids
        iref = findfirst(x -> string(x) == "$id", grad_ref[!, :parameterId])
        @test grad_petab[id] ≈ grad_ref[iref, :value] atol=tol_grad
    end
    # Neural-net parameters
    for nid in keys(model.nn)
        pid = Symbol("p_$nid")
        iref = findall(startswith.(string.(grad_ref[!, :parameterId]), "$(nid)_"))
        @test all(.≈(grad_petab[pid], grad_ref[iref, :value]; atol=tol_grad))
    end
    return nothing
end
