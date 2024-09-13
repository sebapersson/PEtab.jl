using Catalyst
using PEtab
using OrdinaryDiffEq
using Test
using Catalyst
using DataFrames

@testset "Test util functions" begin
    #=
        Test ability to retrive model parameters for specific model conditions
    =#
    # Model without pre-eq or condition specific parameters
    path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    model = PEtabModel(path_yaml, build_julia_files=true, write_to_file=false, verbose=false)
    prob = PEtabODEProblem(model; verbose=false)
    x = prob.xnominal_transformed .* 0.9
    cost = prob.nllh(x)
    res = PEtabOptimisationResult(:Fides,
                                  Vector{Vector{Float64}}(undef, 0),
                                  Vector{Float64}(undef, 0),
                                  10,
                                  cost,
                                  x ./ 0.9,
                                  x,
                                  prob.xnames,
                                  true,
                                  10.0)
    c_id = :model1_data1
    @unpack u0, p = prob.model_info.simulation_info.odesols[c_id].prob
    u0_test = get_u0(res, prob; retmap=false)
    p_test = get_ps(res, prob; retmap=false)
    oprob, _ = get_odeproblem(res, prob)
    sol = get_odesol(res, prob)
    @test all(u0_test .== u0)
    @test all(p_test == p)
    @test all(oprob.u0 .== u0)
    @test all(oprob.p == p)
    @test all(sol.prob.u0 .== u0)
    @test all(sol.prob.p == p)

    # Beer model
    path_yaml = joinpath(@__DIR__, "published_models", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
    prob = PEtabODEProblem(model; verbose=false, sparse_jacobian=false,
                           odesolver=ODESolver(Rodas5P(), abstol=1e-10, reltol=1e-10))
    x = prob.xnominal_transformed .* 0.9
    cost = prob.nllh(x)
    res = PEtabOptimisationResult(:Fides,
                                  Vector{Vector{Float64}}(undef, 0),
                                  Vector{Float64}(undef, 0),
                                  10,
                                  cost,
                                  x ./ 0.9,
                                  x,
                                  prob.xnames,
                                  true,
                                  10.0)
    c_id = :typeIDT1_ExpID1
    @unpack u0, p = prob.model_info.simulation_info.odesols[c_id].prob
    u0_test = get_u0(res, prob; cid=c_id, retmap=false)
    p_test = get_ps(res, prob; cid=c_id, retmap=false)
    @test all(u0_test .== u0)
    to_test = Bool[1, 1, 1, 1, 0, 1, 1, 1, 1] # To account for Event variable
    @test all(p[to_test] == p_test[to_test])

    path_yaml = joinpath(@__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
    model = PEtabModel(path_yaml; build_julia_files = true, verbose=false)
    prob = PEtabODEProblem(model, verbose=false)
    x = prob.xnominal_transformed .* 0.9
    cost = prob.nllh(x)
    res = PEtabOptimisationResult(:Fides,
                                  Vector{Vector{Float64}}(undef, 0),
                                  Vector{Float64}(undef, 0),
                                  10,
                                  cost,
                                  x ./ 0.9,
                                  x,
                                  prob.xnames,
                                  true,
                                  10.0)
    c_id = :Dose_01
    preeq_id = :Dose_0
    @unpack u0, p = prob.model_info.simulation_info.odesols[:Dose_0Dose_01].prob
    p_test = get_ps(res.xmin, prob; cid=c_id, retmap=false)
    u0_test = get_u0(res.xmin, prob; cid=c_id, retmap=false, preeq_id=preeq_id)
    @test all(u0_test .== u0)
    @test all(p == p_test)

    # Case where the system is mutated as we have a initial value set in condition. However,
    # get_odeproblem and its functions should return for the non-mutated input system
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
    model = PEtabModel(rs, simulation_conditions, observables, measurements, params;
                       statemap=u0, verbose = false)
    prob = PEtabODEProblem(model; verbose = false)
    prob.nllh(log10.([1.0, 2.0]))
    ode_prob_mutated = prob.model_info.simulation_info.odesols[:c2].prob
    ode_prob, _ = get_odeproblem(log10.([1.0, 2.0]), prob; cid=:c2)
    @test length(ode_prob.p) == 2
    @test all(ode_prob.p .== ode_prob_mutated.p[[1, 3]])
    @test all(ode_prob.p[[2, 1]] .== ode_prob_mutated.u0)
end
