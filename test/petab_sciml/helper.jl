using SciMLBase, Lux, ComponentArrays, PEtab, CSV, DataFrames, YAML,
      OrdinaryDiffEqRosenbrock, SciMLSensitivity, Test
using Catalyst: @unpack
import Random
rng = Random.default_rng()

PROB_CONFIGS = [(grad = :ForwardDiff, split = false, sensealg = :ForwardDiff),
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

function test_model(test_case, petab_prob::PEtabODEProblem)
    @unpack split_over_conditions, gradient_method = petab_prob.probinfo
    @info "Case $(test_case) and gradient method $(gradient_method) and split = $(split_over_conditions)"
    # Reference values
    dirtest = joinpath(@__DIR__, "test_cases", test_case)
    path_solutions = joinpath(dirtest, "solutions.yaml")
    yamlfile = YAML.load_file(path_solutions)
    llh_ref, tol_llh = yamlfile["llh"], yamlfile["tol_llh"]
    tol_grad = yamlfile["tol_grad_llh"]
    gradfile_mech = yamlfile["grad_llh_files"]["mech"]
    gradmech_ref = CSV.read(joinpath(dirtest, gradfile_mech), DataFrame)
    simfile, tol_sim = yamlfile["simulation_files"][1], yamlfile["tol_simulations"]
    simref = CSV.read(joinpath(dirtest, simfile), DataFrame)

    # Get Parameter values
    x = get_x(petab_prob)
    for (netid, nninfo) in petab_prob.model_info.model.nnmodels
        path_h5 = joinpath(dirtest, "petab", "$(netid)_ps.hf5")
        PEtab.set_ps_net!((@view x[netid]), path_h5, nninfo.nn)
    end

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
    for (netid, nninfo) in petab_prob.model_info.model.nnmodels
        grad_test = grad_petab[netid]
        path_ref = joinpath(dirtest, yamlfile["grad_llh_files"][string(netid)])
        grad_ref = deepcopy(grad_test)
        PEtab.set_ps_net!(grad_ref, path_ref, nninfo.nn)
        @test all(.≈(grad_test, grad_ref; atol=tol_grad))
    end
    return nothing
end
