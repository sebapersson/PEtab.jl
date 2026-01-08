#=
    Test that the PEtab util functions return expected results
=#

using Catalyst, DataFrames, FiniteDifferences, ForwardDiff, OrdinaryDiffEqRosenbrock,
    PEtab, Test

function __sum_ps(x, prob)
    ps = get_ps(x, prob; retmap = false)
    return sum(ps)
end

function __sum_u0(x, prob)
    u0 = get_ps(x, prob; retmap = true)
    return sum(last.(u0))
end

function __sum_ode_problem(x, prob)
    oprob, _ = get_odeproblem(x, prob)
    return sum(solve(oprob, Rodas5P(), abstol = 1e-8, reltol = 1e-8, saveat = 1:10:240))
end

function __sum_sol(x, prob)
    sol = get_odesol(x, prob)
    sol = solve(sol.prob, Rodas5P(), abstol = 1e-8, reltol = 1e-8, saveat = 1:10:240)
    return sum(sol)
end

@testset "util functions" begin
    # Test ability to retrieve model parameters for specific model conditions
    # Model without pre-eq or condition specific parameters
    path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x = get_x(prob) .* 0.9
    nllh = prob.nllh(x)
    res = PEtabOptimisationResult(x ./ 0.9, 10.0, x, :Fides, 10, 10.0,
                                  Vector{Vector{Float64}}(undef, 0), Float64[],  true,  nothing)
    @unpack u0, p = prob.model_info.simulation_info.odesols[:model1_data1].prob
    u0_test = get_u0(res, prob; retmap = false)
    p_test = get_ps(res, prob; retmap = false)
    oprob, _ = get_odeproblem(res, prob)
    sol = get_odesol(res, prob)
    @test all(u0_test .== u0)
    @test all(p_test == p)
    @test all(oprob.u0 .== u0)
    @test all(oprob.p == p)
    @test all(sol.prob.u0 .== u0)
    @test all(sol.prob.p == p)
    # Testing the get_system. As the model is SBML a ReactionSystem model should be returned
    solref = get_odesol(x, prob)
    rn, u0, ps, cb = get_system(x, prob)
    @test rn isa Catalyst.ReactionSystem
    osys = convert(ODESystem, rn) |> structural_simplify |> complete
    oprob = ODEProblem(osys, u0, [solref.t[1], solref.t[end]], ps)
    @test oprob.p.tunable == solref.prob.p
    @test oprob.u0 == solref.prob.u0
    # Test throws correctly
    @test_throws ArgumentError get_ps(res, prob; experiment = :e0)

    # Beer model
    path_yaml = joinpath(@__DIR__, "published_models", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model; sparse_jacobian=false,
                           odesolver=ODESolver(Rodas5P(), abstol=1e-10, reltol=1e-10))
    x = get_x(prob) .* 0.9
    nllh = prob.nllh(x)
    res = PEtabOptimisationResult(x ./ 0.9, 10.0, x, :Fides, 10, 10.0,
                                  Vector{Vector{Float64}}(undef, 0), Float64[],  true,  nothing)
    @unpack u0, p = prob.model_info.simulation_info.odesols[:typeIDT1_ExpID1].prob
    u0_test = get_u0(res, prob; condition = :typeIDT1_ExpID1, retmap = false)
    p_test = get_ps(res, prob; condition = :typeIDT1_ExpID1, retmap = false)
    @test all(u0_test .== u0)
    to_test = Bool[1, 1, 1, 1, 0, 1, 1, 1, 1] # To account for Event variable
    @test all(p[to_test] == p_test[to_test])

    # Model with pre-eq simulation
    path_yaml = joinpath(@__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
    model = PEtabModel(path_yaml; build_julia_files = true, verbose=false)
    prob = PEtabODEProblem(model, verbose=false)
    x = get_x(prob) .* 0.9
    nllh = prob.nllh(x)
    res = PEtabOptimisationResult(x ./ 0.9, 10.0, x, :Fides, 10, 10.0,
                                  Vector{Vector{Float64}}(undef, 0), Float64[],  true,  nothing)
    @unpack u0, p = prob.model_info.simulation_info.odesols[:Dose_0Dose_01].prob
    p_test = get_ps(res.xmin, prob; condition = :Dose_0 => :Dose_01, retmap = false)
    u0_test = get_u0(res.xmin, prob; condition = "Dose_0" => "Dose_01", retmap = false)
    @test all(u0_test .== u0)
    @test all(p == p_test)
    oprob, _ = get_odeproblem(res, prob; condition = :Dose_0 => :Dose_01)
    @test all(oprob.u0 .== u0)
    @test all(oprob.p == p_test)
    @test oprob.tspan[end] == prob.model_info.simulation_info.tmaxs[:Dose_0Dose_01]
    @test_throws PEtab.PEtabInputError begin
        oprob, _ = get_odeproblem(res, prob; condition = :Dose_01 => :Dose_01)
    end

    # Case where the system is mutated as we have a initial value set in condition. However,
    # get_odeproblem and its functions should return for the non-mutated input system
    rs = @reaction_network begin
        (k1, k2), X1 <--> X2
    end
    u0 = [:X1 => 1.0]
    @unpack X1 = rs
    observables = PEtabObservable("obs_X1", X1, 0.5)
    par_k1 = PEtabParameter(:k1)
    par_k2 = PEtabParameter(:k2)
    params = [par_k1, par_k2]
    simulation_conditions = [PEtabCondition(:c1, :X2 => 1.0),
                             PEtabCondition(:c2, :X2 => 2.0)]
    m_c1 = DataFrame(simulation_id = "c1", obs_id="obs_X1", time=[1.0, 2.0, 3.0], measurement=[1.1, 1.2, 1.3])
    m_c2 = DataFrame(simulation_id = "c2", obs_id="obs_X1", time=[1.0, 2.0, 3.0], measurement=[1.2, 1.4, 1.6])
    measurements = vcat(m_c1, m_c2)
    model = PEtabModel(rs, observables, measurements, params; speciemap=u0,
                       simulation_conditions = simulation_conditions)
    prob = PEtabODEProblem(model; verbose = false)
    prob.nllh(log10.([1.0, 2.0]))
    oprob_mutated = prob.model_info.simulation_info.odesols[:c2].prob
    oprob, _ = get_odeproblem(log10.([1.0, 2.0]), prob; condition =:c2)
    @test length(oprob.p) == 2
    @test all(oprob.p .== oprob_mutated.p[[1, 3]])
    @test all(oprob.p[[2, 1]] .== oprob_mutated.u0)
    # Test that get_system correctly returns a ReactionSystem
    rn, u0, ps, _ = get_system(log10.([1.0, 2.0]), prob; condition = :c2)
    @test rn isa Catalyst.ReactionSystem
    osys = convert(ODESystem, rn) |> structural_simplify |> complete
    oprob_sys = ODEProblem(osys, u0, (0.0, 1.0), ps)
    @test oprob_sys.p.tunable == oprob.p
    @test oprob_sys.u0 == oprob.u0
    @test_throws PEtab.PEtabInputError begin
        oprob, _ = get_odeproblem(res, prob; condition = :c3)
    end

    # Verify Dual numbers can be propagated through get functions
    path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    prob = PEtabModel(path_yaml) |>
        PEtabODEProblem
    x = get_x(prob) .* 0.9
    _sum_ps = (x) -> __sum_ps(x, prob)
    _sum_u0 = (x) -> __sum_u0(x, prob)
    _sum_ode_problem = (x) -> __sum_ode_problem(x, prob)
    _sum_sol = (x) -> __sum_sol(x, prob)
    # Reference via Finite-diff
    grad_ps_ref = FiniteDifferences.grad(central_fdm(5, 1), _sum_ps, x)[1]
    grad_u0_ref = FiniteDifferences.grad(central_fdm(5, 1), _sum_u0, x[:])[1]
    grad_prob_ref = FiniteDifferences.grad(central_fdm(5, 1), _sum_ode_problem, x)[1]
    grad_sol_ref = FiniteDifferences.grad(central_fdm(5, 1), _sum_sol, x[:])[1]
    grad_ps = ForwardDiff.gradient(_sum_ps, x)
    grad_u0 = ForwardDiff.gradient(_sum_u0, x[:])
    grad_prob = ForwardDiff.gradient(_sum_ode_problem, x)
    grad_sol = ForwardDiff.gradient(_sum_sol, x[:])
    @test all(.≈(grad_ps_ref, grad_ps, atol = 1e-3))
    @test all(.≈(grad_u0_ref, grad_u0, atol = 1e-3))
    @test all(.≈(grad_prob_ref, grad_prob, atol = 1e-3))
    @test all(.≈(grad_sol_ref, grad_sol, atol = 1e-3))

    # PEtab v2 problems use experiment for time-course
    path_yaml = joinpath(@__DIR__, "petab_v2_testsuite", "0002", "_0002.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x = get_x(prob)
    _ = prob.nllh(x)
    ode1 = prob.model_info.simulation_info.odesols[:e1_c0].prob
    ode2 = prob.model_info.simulation_info.odesols[:e2_c1].prob
    # Test all the get functions
    u0_e1 = get_u0(x, prob; retmap = false)
    u0_e2 = get_u0(x, prob; experiment = :e2, retmap = false)
    ps_e1 = get_ps(x, prob; retmap = false)
    ps_e2 = get_ps(x, prob; experiment = :e2, retmap = false)
    ode_e1, _ = get_odeproblem(x, prob; experiment = :e1)
    sol_e1 = get_odesol(x, prob; experiment = :e1)
    @test ode1.u0 == u0_e1
    @test ode2.u0 == u0_e2
    @test ode1.p == ps_e1
    @test ode2.p == ps_e2
    @test ode_e1.p == ode1.p
    @test ode_e1.u0 == ode1.u0
    @test ode_e1.p == sol_e1.prob.p
    @test ode_e1.u0 == sol_e1.prob.u0
    # Test error handling
    @test_throws ArgumentError get_ps(x, prob; condition = :e1)
    @test_throws ArgumentError get_ps(x, prob; condition = :e1, experiment = :e1)
    @test_throws PEtab.PEtabInputError get_ps(x, prob; experiment = :e5)

    # PEtab v2 problem with pre-equlibration
    path_yaml = joinpath(@__DIR__, "petab_v2_testsuite", "0009", "_0009.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x = get_x(prob)
    _ = prob.nllh(x)
    ode = prob.model_info.simulation_info.odesols[:e0_preeq_c0e0_c0].prob
    @test ode.p == get_ps(x, prob; retmap = false)
    @test all(.≈(ode.u0, get_u0(x, prob; experiment = :e0, retmap = false), atol = 1e-8))

    path_yaml = joinpath(@__DIR__, "petab_v2_testsuite", "0001", "_0001.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x = get_x(prob)
    @test_throws PEtab.PEtabInputError get_ps(x, prob; experiment = :e0)
end
