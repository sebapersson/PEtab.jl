using Catalyst
using PEtab
using OrdinaryDiffEq
using Test


@testset "Extract simulation parameters" begin
    #=
    Test ability to retrive model parameters for specific model conditions
    =#
    # Model without pre-eq or condition specific parameters
    path_yaml = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    petab_model = PEtabModel(path_yaml, build_julia_files=true, write_to_file=false, verbose=false)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)
    θ = petab_problem.θ_nominalT .* 0.9
    cost = petab_problem.compute_cost(θ)
    res = PEtabOptimisationResult(:Fides,
                                  Vector{Vector{Float64}}(undef, 0),
                                  Vector{Float64}(undef, 0),
                                  10,
                                  cost,
                                  θ ./ 0.9,
                                  θ,
                                  petab_problem.θ_names,
                                  true,
                                  10.0)
    c_id = :model1_data1
    @unpack u0, p = petab_problem.simulation_info.ode_sols[c_id].prob
    u0_test = get_u0(res, petab_problem; retmap=false)
    p_test = get_ps(res, petab_problem; retmap=false)
    @test all(u0_test .== u0)
    @test all(p == p_test)

    # Beer model
    path_yaml = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
    petab_problem = PEtabODEProblem(petab_model, verbose=false, 
                                    ode_solver=ODESolver(Rodas5P()),
                                    sparse_jacobian=false)
    θ = petab_problem.θ_nominalT .* 0.9
    cost = petab_problem.compute_cost(θ)
    res = PEtabOptimisationResult(:Fides,
                                  Vector{Vector{Float64}}(undef, 0),
                                  Vector{Float64}(undef, 0),
                                  10,
                                  cost,
                                  θ ./ 0.9,
                                  θ,
                                  petab_problem.θ_names,
                                  true,
                                  10.0)
    c_id = :typeIDT1_ExpID1
    @unpack u0, p = petab_problem.simulation_info.ode_sols[c_id].prob
    u0_test = get_u0(res, petab_problem; condition_id=c_id, retmap=false)
    p_test = get_ps(res, petab_problem; condition_id=c_id, retmap=false)
    @test all(u0_test .== u0)
    @test all(p[2:end] == p_test[2:end]) # 2:end to account for event variable

    # Brannmark model
    path_yaml = joinpath(@__DIR__, "Test_ll", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)
    θ = petab_problem.θ_nominalT .* 0.9
    cost = petab_problem.compute_cost(θ)
    res = PEtabOptimisationResult(:Fides,
                                  Vector{Vector{Float64}}(undef, 0),
                                  Vector{Float64}(undef, 0),
                                  10,
                                  cost,
                                  θ ./ 0.9,
                                  θ,
                                  petab_problem.θ_names,
                                  true,
                                  10.0)
    c_id = :Dose_01
    pre_eq_id = :Dose_0
    @unpack u0, p = petab_problem.simulation_info.ode_sols[:Dose_0Dose_01].prob
    p_test = get_ps(res, petab_problem; condition_id=c_id, retmap=false)
    u0_test = get_u0(res, petab_problem; condition_id=c_id, retmap=false, pre_eq_id=pre_eq_id)
    @test all(u0_test .== u0)
    @test all(p == p_test)
end