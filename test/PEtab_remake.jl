using PEtab
using Test
using OrdinaryDiffEq
using Sundials
using CSV
using ForwardDiff
using LinearAlgebra


function test_petab_remake(petab_model::PEtabModel, parameters_change, what_check)

    if :GradientForwardDiff ∈ what_check
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), chunksize=2, verbose=false)
        petab_problem2 = remake_PEtab_problem(petab_problem1, parameters_change)

        iMatch = findall(in(petab_problem2.θ_names), petab_problem1.θ_names)
        g1ForwardDiff = zeros(length(petab_problem1.θ_nominalT))
        g2ForwardDiff = zeros(length(petab_problem2.θ_nominalT))
        petab_problem1.compute_gradient!(g1ForwardDiff, petab_problem1.θ_nominalT)
        petab_problem2.compute_gradient!(g2ForwardDiff, petab_problem2.θ_nominalT)
        @test norm(g1ForwardDiff[iMatch] - g2ForwardDiff) / length(g2ForwardDiff) ≤ 1e-3

        # Also test the nllh gradient 
        petab_problem1.compute_gradient_nllh!(g1ForwardDiff, petab_problem1.θ_nominalT)
        petab_problem2.compute_gradient_nllh!(g2ForwardDiff, petab_problem2.θ_nominalT)
        @test norm(g1ForwardDiff[iMatch] - g2ForwardDiff) / length(g2ForwardDiff) ≤ 1e-3
    end

    if :GradientForwardEquations ∈ what_check
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), chunksize=2, gradient_method=:ForwardEquations, sensealg=:ForwardDiff, verbose=false)
        petab_problem2 = remake_PEtab_problem(petab_problem1, parameters_change)

        iMatch = findall(in(petab_problem2.θ_names), petab_problem1.θ_names)
        g1ForwardEq = zeros(length(petab_problem1.θ_nominalT))
        g2ForwardEq = zeros(length(petab_problem2.θ_nominalT))
        petab_problem1.compute_gradient!(g1ForwardEq, petab_problem1.θ_nominalT)
        petab_problem2.compute_gradient!(g2ForwardEq, petab_problem2.θ_nominalT)
        @test norm(g1ForwardEq[iMatch] - g2ForwardEq) / length(g2ForwardEq) ≤ 1e-3
    end

    if :GaussNewton ∈ what_check
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), chunksize=2, gradient_method=:ForwardDiff, hessian_method=:GaussNewton, sensealg=:ForwardDiff, verbose=false)
        petab_problem2 = remake_PEtab_problem(petab_problem1, parameters_change)

        iMatch = findall(in(petab_problem2.θ_names), petab_problem1.θ_names)
        _h1GN = zeros(length(petab_problem1.θ_nominalT), length(petab_problem1.θ_nominalT))
        h1GN = zeros(length(petab_problem2.θ_nominalT), length(petab_problem2.θ_nominalT))
        h2GN = zeros(length(petab_problem2.θ_nominalT), length(petab_problem2.θ_nominalT))
        petab_problem1.compute_hessian!(_h1GN, petab_problem1.θ_nominalT)
        petab_problem2.compute_hessian!(h2GN, petab_problem2.θ_nominalT)
        for (i1, i2) in pairs(iMatch)
            for (j1, j2) in pairs(iMatch)
                h1GN[i1, j1] = _h1GN[i2, j2]
            end
        end
        @test norm(h1GN - h2GN) ≤ 1e-3
    end

    if :Hessian ∈ what_check
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), chunksize=2, hessian_method=:ForwardDiff, verbose=false)
        petab_problem2 = remake_PEtab_problem(petab_problem1, parameters_change)

        iMatch = findall(in(petab_problem2.θ_names), petab_problem1.θ_names)
        _h1 = zeros(length(petab_problem1.θ_nominalT), length(petab_problem1.θ_nominalT))
        h1 = zeros(length(petab_problem2.θ_nominalT), length(petab_problem2.θ_nominalT))
        h2 = zeros(length(petab_problem2.θ_nominalT), length(petab_problem2.θ_nominalT))
        petab_problem1.compute_cost(petab_problem1.θ_nominalT) # ensure to allocate certain objects
        petab_problem1.compute_hessian!(_h1, petab_problem1.θ_nominalT)
        petab_problem2.compute_hessian!(h2, petab_problem2.θ_nominalT)
        for (i1, i2) in pairs(iMatch)
            for (j1, j2) in pairs(iMatch)
                h1[i1, j1] = _h1[i2, j2]
            end
        end
        @test norm(h1 - h2) ≤ 1e-3
    end

    if :FIM ∈ what_check
        petab_problem1 = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), chunksize=2, hessian_method=:ForwardDiff, verbose=false)
        petab_problem2 = remake_PEtab_problem(petab_problem1, parameters_change)

        iMatch = findall(in(petab_problem2.θ_names), petab_problem1.θ_names)
        _FIM1 = zeros(length(petab_problem1.θ_nominalT), length(petab_problem1.θ_nominalT))
        FIM1 = zeros(length(petab_problem2.θ_nominalT), length(petab_problem2.θ_nominalT))
        FIM2 = zeros(length(petab_problem2.θ_nominalT), length(petab_problem2.θ_nominalT))
        petab_problem1.compute_cost(petab_problem1.θ_nominalT) # ensure to allocate certain objects
        petab_problem1.compute_FIM!(_FIM1, petab_problem1.θ_nominalT)
        petab_problem2.compute_FIM!(FIM2, petab_problem2.θ_nominalT)
        for (i1, i2) in pairs(iMatch)
            for (j1, j2) in pairs(iMatch)
                FIM1[i1, j1] = _FIM1[i2, j2]
            end
        end
        @test norm(FIM1 - FIM2) ≤ 1e-3
    end

end


# Test for Boehm (no steady-state simulations)
@info "Testing remake for Boehm model"
petab_model = PEtabModel(joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"), build_julia_files=false, verbose=false)
parameters_change1 = Dict(:k_imp_hetero => 0.0163679184468,
                        :k_exp_homo => 0.006170228086381,
                        :k_phos => 15766.5070195731)
parameters_change2 = Dict(:k_imp_hetero => "estimate",
                        :k_exp_homo => 0.006170228086381,
                        :k_phos => "estimate")
parameters_change3 = Dict(:sd_pSTAT5A_rel => 3.85261197844677)
parameters_change4 = Dict(:sd_pSTAT5A_rel => 3.85261197844677,
                          :k_exp_homo => 0.006170228086381)
@testset "Test PEtab remake : Boehm" begin
    test_petab_remake(petab_model, parameters_change1, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
    test_petab_remake(petab_model, parameters_change2, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
    test_petab_remake(petab_model, parameters_change3, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
    test_petab_remake(petab_model, parameters_change4, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
end

# Test for Brannmark (has steady state simulations)
@info "Testing remake for Brannmark model"
petab_model = PEtabModel(joinpath(@__DIR__, "Test_ll", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml"), build_julia_files=true, verbose=false)
parameters_change1 = Dict(:k1a => 0.177219477727669,
                          :k22 => 666.8355739795,
                          :km2 => 1.15970741690448)
parameters_change2 = Dict(:k1a => "estimate",
                          :k22 => 666.8355739795,
                          :km2 => "estimate")
parameters_change3 = Dict(:sigmaY2Step => 5.15364156671777)
parameters_change4 = Dict(:sigmaY2Step => 5.15364156671777,
                          :k22 => 666.8355739795)
@testset "Test PEtab remake : Brannmark" begin
    test_petab_remake(petab_model, parameters_change1, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    test_petab_remake(petab_model, parameters_change2, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    test_petab_remake(petab_model, parameters_change3, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    test_petab_remake(petab_model, parameters_change4, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
end