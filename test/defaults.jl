#=
    Test that default options (user does not provide ODE-solver and gradient-method)
    is handled correctly.
=#

using PEtab, SciMLSensitivity, OrdinaryDiffEq, Sundials, Test

@testset "Test default options" begin
    # Check that we get correct default setting
    path = joinpath(@__DIR__, "published_models", "Bachmann_MSB2011", "Bachmann_MSB2011.yaml")
    model = PEtabModel(path; verbose=false, build_julia_files=false, write_to_file = false)
    prob = PEtabODEProblem(model)
    @test prob.probinfo.gradient_method === :ForwardDiff
    @test prob.probinfo.hessian_method === :GaussNewton
    @test prob.probinfo.solver.solver isa QNDF
    prob = PEtabODEProblem(model, reuse_sensitivities=true, verbose=false)
    @test prob.probinfo.gradient_method === :ForwardEquations
    @test prob.probinfo.hessian_method === :GaussNewton
    @test prob.probinfo.solver.solver isa QNDF
    @test prob.probinfo.split_over_conditions == false

    path = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    model = PEtabModel(path)
    prob = PEtabODEProblem(model)
    @test prob.probinfo.gradient_method === :ForwardDiff
    @test prob.probinfo.hessian_method === :ForwardDiff
    @test prob.probinfo.solver.solver isa Rodas5P
    @test prob.probinfo.split_over_conditions == false
    # Test we can compute gradient for a given input
    @test_throws PEtab.PEtabInputError PEtabODEProblem(model;  odesolver = ODESolver(CVODE_BDF()))
    prob = PEtabODEProblem(model; odesolver = ODESolver(CVODE_BDF()), odesolver_gradient = ODESolver(QNDF()))
    @test prob.probinfo.solver_gradient.solver isa QNDF

    # Here split_over_conditions = true as this model has many condition specific parameters
    path = joinpath(@__DIR__, "published_models", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    model = PEtabModel(path; verbose=false, build_julia_files=false, write_to_file = false)
    prob = PEtabODEProblem(model)
    @test prob.probinfo.gradient_method === :ForwardDiff
    @test prob.probinfo.hessian_method === :ForwardDiff
    @test prob.probinfo.solver.solver isa Rodas5P
    @test prob.probinfo.split_over_conditions == true

    # Adjoint sensitivity analysis model
    path = joinpath(@__DIR__, "published_models", "Smith_BMCSystBiol2013", "Smith_BMCSystBiol2013.yaml")
    model = PEtabModel(path; verbose=false, build_julia_files=false, write_to_file = false)
    prob = PEtabODEProblem(model; verbose = false)
    @test prob.probinfo.gradient_method === :Adjoint
    @test prob.probinfo.hessian_method === :GaussNewton
    @test prob.probinfo.solver.solver isa CVODE_BDF
    @test prob.probinfo.split_over_conditions == false
    prob = PEtabODEProblem(model; gradient_method=:ForwardDiff, verbose=false)
    @test prob.probinfo.solver.solver isa KenCarp4
    @test prob.probinfo.sparse_jacobian == false
end
