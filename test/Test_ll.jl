#=
    Check the accruacy of the PeTab importer by checking the log-likelihood value against known values for several
    published models. Also check gradients for selected models using FiniteDifferences package
=#


using PEtab
using Test
using OrdinaryDiffEq
using Zygote 
using SciMLSensitivity
using CSV
using ForwardDiff
using LinearAlgebra
using FiniteDifferences
using Sundials


# Used to test cost-value at the nominal parameter value
function test_loglikelihood(petab_model::PEtabModel,
                                referenceValue::Float64,
                                solverOptions; atol=1e-3,
                                verbose=false, checkZygote=true)

    model_name = petab_model.model_name
    @info "Model : $model_name"
    petab_problem1 = PEtabODEProblem(petab_model, ode_solver=solverOptions, cost_method=:Standard, verbose=verbose)
    petab_problem2 = PEtabODEProblem(petab_model, ode_solver=solverOptions, cost_method=:Zygote, verbose=verbose)
    cost = petab_problem1.compute_cost(petab_problem1.θ_nominalT)
    @test cost ≈ referenceValue atol = atol

    if checkZygote == true
        costZygote = petab_problem2.compute_cost(petab_problem1.θ_nominalT)
        @test costZygote ≈ referenceValue atol = atol
    end
end


function test_gradient_finite_differences(petab_model::PEtabModel, solverOptions;
                                       solverGradientOptions=nothing,
                                       checkForwardEquations::Bool=false,
                                       checkAdjoint::Bool=false,
                                       testtol::Float64=1e-3,
                                       sensealg_ss=SteadyStateAdjoint(),
                                       sensealgAdjoint=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                       ss_options=nothing,
                                       onlyCheckAutoDiff::Bool=false,
                                       split_over_conditions=false)

    if isnothing(solverGradientOptions)
        solverGradientOptions = deepcopy(solverOptions)
    end

    # Testing the gradient via finite differences
    petab_problem1 = PEtabODEProblem(petab_model, ode_solver=solverOptions,
                                          gradient_method=:ForwardDiff,
                                          split_over_conditions=split_over_conditions,
                                          ss_solver=ss_options,
                                          verbose=false, 
                                          sparse_jacobian=false)
    θ_use = petab_problem1.θ_nominalT
    gradientFinite = FiniteDifferences.grad(central_fdm(5, 1), petab_problem1.compute_cost, θ_use)[1]
    gradientForward = zeros(length(θ_use))
    petab_problem1.compute_gradient!(gradientForward, θ_use)
    @test norm(gradientFinite - gradientForward) ≤ testtol

    if checkForwardEquations == true
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=solverOptions,
                                             gradient_method=:ForwardEquations, sensealg=:ForwardDiff,
                                             ss_solver=ss_options,
                                             verbose=false, 
                                             sparse_jacobian=false)
        gradientForwardEquations1 = zeros(length(θ_use))
        petab_problem1.compute_gradient!(gradientForwardEquations1, θ_use)
        @test norm(gradientFinite - gradientForwardEquations1) ≤ testtol

        if onlyCheckAutoDiff == false
            petab_problem2 = PEtabODEProblem(petab_model, ode_solver=solverOptions, ode_solver_gradient=solverGradientOptions,
                                                 gradient_method=:ForwardEquations, sensealg=ForwardSensitivity(),
                                                 ss_solver=ss_options,
                                                 verbose=false, 
                                                 sparse_jacobian=false)
            gradientForwardEquations2 = zeros(length(θ_use))
            petab_problem2.compute_gradient!(gradientForwardEquations2, θ_use)
            @test norm(gradientFinite - gradientForwardEquations2) ≤ testtol
        end
    end

    if checkAdjoint == true
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=solverOptions, ode_solver_gradient=solverGradientOptions,
                                             gradient_method=:Adjoint, sensealg=sensealgAdjoint, sensealg_ss=sensealg_ss,
                                             split_over_conditions=split_over_conditions,
                                             ss_solver=ss_options,
                                             verbose=false, 
                                             sparse_jacobian=false)
        gradient_adjoint = zeros(length(θ_use))
        petab_problem1.compute_gradient!(gradient_adjoint, θ_use)
        @test norm(gradientFinite - gradient_adjoint) ≤ testtol
    end
end


# Bachman model
path_yaml = joinpath(@__DIR__, "Test_ll", "Bachmann_MSB2011", "Bachmann_MSB2011.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
test_loglikelihood(petab_model, -418.40573341425295, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12))
test_gradient_finite_differences(petab_model, ODESolver(Rodas4P(), abstol=1e-9, reltol=1e-9),
                             solverGradientOptions=ODESolver(CVODE_BDF(), abstol=1e-9, reltol=1e-9),
                             checkForwardEquations=true, checkAdjoint=true, testtol=1e-2)

# Beer model - Numerically challenging gradient as we have callback time triggering parameters to
# estimate. Splitting over conditions spped up hessian computations with factor 48
path_yaml = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
test_loglikelihood(petab_model, -58622.9145631413, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12), checkZygote=false)
test_gradient_finite_differences(petab_model, ODESolver(Rodas4P(), abstol=1e-8, reltol=1e-8), testtol=1e-1, onlyCheckAutoDiff=true, checkForwardEquations=true, split_over_conditions=true)

# Brännmark model. Model has pre-equlibration criteria so here we test all gradients. Challenging to compute gradients.
path_yaml = joinpath(@__DIR__, "Test_ll", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
test_loglikelihood(petab_model, 141.889113770537, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12))
test_gradient_finite_differences(petab_model, ODESolver(Rodas5(), abstol=1e-8, reltol=1e-8), onlyCheckAutoDiff=true, checkForwardEquations=true, testtol=2e-3)

# Crauste model. The model is numerically challanging and computing a gradient via Finite-differences is not possible
path_yaml = joinpath(@__DIR__, "Test_ll", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
test_loglikelihood(petab_model, 190.96521897435176, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12), atol=1e-2)

# Fujita model. Challangeing to compute accurate gradients
path_yaml = joinpath(@__DIR__, "Test_ll", "Fujita_SciSignal2010", "Fujita_SciSignal2010.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
test_loglikelihood(petab_model, -53.08377736998929, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12))

# Iseense - tricky model with pre-eq criteria and priors
path_yaml = joinpath(@__DIR__, "Test_ll", "Isensee_JCB2018", "Isensee_JCB2018.yaml")
petab_model = PEtabModel(path_yaml, verbose=true, build_julia_files=true)
test_loglikelihood(petab_model, 3949.375966548649 + 4.45299970460275, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12), checkZygote=false)

# Sneyd model - Test against World problem by wrapping inside function
function testSneyd()
    path_yaml = joinpath(@__DIR__, "Test_ll", "Sneyd_PNAS2002", "Sneyd_PNAS2002.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
    test_loglikelihood(petab_model, -319.79177818768756, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12))
end
testSneyd()

# Zheng - has SBML functions 
path_yaml = joinpath(@__DIR__, "Test_ll", "Zheng_PNAS2012", "Zheng_PNAS2012.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
test_loglikelihood(petab_model, -278.33353271001477, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12))
