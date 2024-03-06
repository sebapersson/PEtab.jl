using Catalyst
using PEtab
using OrdinaryDiffEq
using Test
using Catalyst
using DataFrames


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
    odeprob, _, _ = get_odeproblem(res, petab_problem)
    sol = get_odesol(res, petab_problem)
    @test all(u0_test .== u0)
    @test all(p_test == p)
    @test all(odeprob.u0 .== u0)
    @test all(odeprob.p == p)
    @test all(sol.prob.u0 .== u0)
    @test all(sol.prob.p == p)
    # Test runtime and accuracy
    time1, acc1 = compute_runtime_accuracy(θ, petab_problem, Rodas5())
    time2, acc2 = compute_runtime_accuracy(θ, petab_problem, KenCarp4())
    @test acc1 < acc2
    @test time1 < 1.0
    @test time2 < 1.0

    # Beer model
    path_yaml = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
    petab_problem = PEtabODEProblem(petab_model, verbose=false,
                                    ode_solver=ODESolver(Rodas5P(), abstol=1e-10, reltol=1e-10),
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
    to_test = Bool[1, 1, 1, 1, 0, 1, 1, 1, 1] # To account for Event variable
    @test all(p[to_test] == p_test[to_test])

    # Test solve all conditions
    θ = petab_problem.θ_nominalT[:]
    odesols_test, could_solve = solve_all_conditions(θ, petab_problem, Rodas5P(); save_at_observed_t=false, abstol=1e-10, reltol=1e-10)
    petab_problem.compute_cost(θ)
    odesols_ref = petab_problem.simulation_info.ode_sols
    for id in keys(odesols_ref)
        diff = abs.(odesols_test[id][:, end] - odesols_ref[id][:, end])
        @test all(diff .< 1e-12)
    end
    odesols_test, could_solve = PEtab.solve_all_conditions(θ, petab_problem, Rodas5P(); n_timepoints_save=100)
    for id in keys(odesols_ref)
        @test length(odesols_test[id].t) == 100
    end

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
    p_test = get_ps(res.xmin, petab_problem; condition_id=c_id, retmap=false)
    u0_test = get_u0(res.xmin, petab_problem; condition_id=c_id, retmap=false, pre_eq_id=pre_eq_id)
    @test all(u0_test .== u0)
    @test all(p == p_test)
    # Test solve all conditions for model with pre-eq
    θ = petab_problem.θ_nominalT[:]
    odesols_test, could_solve = PEtab.solve_all_conditions(θ, petab_problem, Rodas5P(); save_at_observed_t=true)
    petab_problem.compute_cost(θ)
    odesols_ref = petab_problem.simulation_info.ode_sols
    for id in keys(odesols_ref)
        diff = abs.(odesols_test[id][:, end] - odesols_ref[id][:, end])
        @test all(diff .< 1e-12)
    end

    # Case where the system is mutated as we have a initial value set in condition. However, get_odeproblem
    # and its functions should return for the non-mutated input system
    rs = @reaction_network begin
        (k1, k2), X1 <--> X2
    end
    u0 = [:X1 => 1.0]
    @unpack X1 = rs
    obs_X1 = PEtabObservable(X1, 0.5)
    observables = Dict("obs_X1" => obs_X1)
    par_k1 = PEtabParameter(:k1)
    par_k2 = PEtabParameter(:k2)
    params = [par_k1, par_k2]
    c1 = Dict(:X2 => 1.0)
    c2 = Dict(:X2 => 2.0)
    simulation_conditions = Dict("c1" => c1, "c2" => c2)
    m_c1 = DataFrame(simulation_id = "c1", obs_id="obs_X1", time=[1.0, 2.0, 3.0], measurement=[1.1, 1.2, 1.3])
    m_c2 = DataFrame(simulation_id = "c2", obs_id="obs_X1", time=[1.0, 2.0, 3.0], measurement=[1.2, 1.4, 1.6])
    measurements = vcat(m_c1, m_c2)
    petab_model = PEtabModel(rs, simulation_conditions, observables, measurements, params; state_map=u0)
    petab_problem = PEtabODEProblem(petab_model)
    petab_problem.compute_cost(log10.([1.0, 2.0]))
    ode_prob_mutated = petab_problem.simulation_info.ode_sols[:c2].prob
    ode_prob, _, _ = get_odeproblem(log10.([1.0, 2.0]), petab_problem; condition_id=:c2)
    @test length(ode_prob.p) == 2
    @test all(ode_prob.p .== ode_prob_mutated.p[1:2])
    @test all(ode_prob.p .== ode_prob_mutated.u0)
end
