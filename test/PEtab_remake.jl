using PEtab
using Test
using OrdinaryDiffEq
using Sundials
using SciMLSensitivity
using CSV
using DataFrames
using ForwardDiff
using LinearAlgebra


function testPEtabRemake(petabModel::PEtabModel, parametersChange, whatCheck)
    
    if :GradientForwardDiff ∈ whatCheck
        petabProblem1 = setupPEtabODEProblem(petabModel, getODESolverOptions(Rodas5P()), chunkSize=2, verbose=false)
        petabProblem2 = remakePEtabProblem(petabProblem1, parametersChange)

        iMatch = findall(in(petabProblem2.θ_estNames), petabProblem1.θ_estNames)
        g1ForwardDiff = zeros(length(petabProblem1.θ_nominalT))
        g2ForwardDiff = zeros(length(petabProblem2.θ_nominalT))
        petabProblem1.computeGradient!(g1ForwardDiff, petabProblem1.θ_nominalT) 
        petabProblem2.computeGradient!(g2ForwardDiff, petabProblem2.θ_nominalT) 
        @test norm(g1ForwardDiff[iMatch] - g2ForwardDiff) / length(g2ForwardDiff) ≤ 1e-3
    end

    if :GradientForwardEquations ∈ whatCheck
        petabProblem1 = setupPEtabODEProblem(petabModel, getODESolverOptions(Rodas5P()), chunkSize=2, gradientMethod=:ForwardEquations, sensealg=:ForwardDiff, verbose=false)
        petabProblem2 = remakePEtabProblem(petabProblem1, parametersChange)

        iMatch = findall(in(petabProblem2.θ_estNames), petabProblem1.θ_estNames)
        g1ForwardEq = zeros(length(petabProblem1.θ_nominalT))
        g2ForwardEq = zeros(length(petabProblem2.θ_nominalT))
        petabProblem1.computeGradient!(g1ForwardEq, petabProblem1.θ_nominalT) 
        petabProblem2.computeGradient!(g2ForwardEq, petabProblem2.θ_nominalT) 
        @test norm(g1ForwardEq[iMatch] - g2ForwardEq) / length(g2ForwardEq) ≤ 1e-3
    end

    if :GaussNewton ∈ whatCheck
        petabProblem1 = setupPEtabODEProblem(petabModel, getODESolverOptions(Rodas5P()), chunkSize=2, hessianMethod=:GaussNewton, sensealg=:ForwardDiff, verbose=false)
        petabProblem2 = remakePEtabProblem(petabProblem1, parametersChange)
        
        iMatch = findall(in(petabProblem2.θ_estNames), petabProblem1.θ_estNames)
        _h1GN = zeros(length(petabProblem1.θ_nominalT), length(petabProblem1.θ_nominalT))
        h1GN = zeros(length(petabProblem2.θ_nominalT), length(petabProblem2.θ_nominalT))
        h2GN = zeros(length(petabProblem2.θ_nominalT), length(petabProblem2.θ_nominalT))
        petabProblem1.computeHessian!(_h1GN, petabProblem1.θ_nominalT)
        petabProblem2.computeHessian!(h2GN, petabProblem2.θ_nominalT)
        for (i1, i2) in pairs(iMatch)
            for (j1, j2) in pairs(iMatch)
                h1GN[i1, j1] = _h1GN[i2, j2]
            end
        end
        @test norm(h1GN - h2GN) ≤ 1e-3
    end

    if :Hessian ∈ whatCheck
        petabProblem1 = setupPEtabODEProblem(petabModel, getODESolverOptions(Rodas5P()), chunkSize=2, hessianMethod=:ForwardDiff, verbose=false)
        petabProblem2 = remakePEtabProblem(petabProblem1, parametersChange)
        
        iMatch = findall(in(petabProblem2.θ_estNames), petabProblem1.θ_estNames)
        _h1 = zeros(length(petabProblem1.θ_nominalT), length(petabProblem1.θ_nominalT))
        h1 = zeros(length(petabProblem2.θ_nominalT), length(petabProblem2.θ_nominalT))
        h2 = zeros(length(petabProblem2.θ_nominalT), length(petabProblem2.θ_nominalT))
        petabProblem1.computeCost(petabProblem1.θ_nominalT) # ensure to allocate certain objects 
        petabProblem1.computeHessian!(_h1, petabProblem1.θ_nominalT)
        petabProblem2.computeHessian!(h2, petabProblem2.θ_nominalT)
        for (i1, i2) in pairs(iMatch)
            for (j1, j2) in pairs(iMatch)
                h1[i1, j1] = _h1[i2, j2]
            end
        end
        @test norm(h1 - h2) ≤ 1e-3
    end

end


# Test for Boehm (no steady-state simulations)
@info "Testing remake for Boehm model"
petabModel = readPEtabModel(joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"), forceBuildJuliaFiles=false, verbose=false)
parametersChange1 = Dict(:k_imp_hetero => 0.0163679184468, 
                        :k_exp_homo => 0.006170228086381, 
                        :k_phos => 15766.5070195731)             
parametersChange2 = Dict(:k_imp_hetero => "estimate", 
                        :k_exp_homo => 0.006170228086381, 
                        :k_phos => "estimate")            
parametersChange3 = Dict(:sd_pSTAT5A_rel => 3.85261197844677)
parametersChange4 = Dict(:sd_pSTAT5A_rel => 3.85261197844677, 
                         :k_exp_homo => 0.006170228086381)
@testset "Test PEtab remake : Boehm" begin                          
    testPEtabRemake(petabModel, parametersChange1, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian])
    testPEtabRemake(petabModel, parametersChange2, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian])
    testPEtabRemake(petabModel, parametersChange3, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian])
    testPEtabRemake(petabModel, parametersChange4, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian])
end

# Test for Brannmark (has steady state simulations)
@info "Testing remake for Brannmark model"
petabModel = readPEtabModel(joinpath(@__DIR__, "Test_ll", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml"), forceBuildJuliaFiles=false, verbose=false)
parametersChange1 = Dict(:k1a => 0.177219477727669, 
                        :k22 => 666.8355739795, 
                        :km2 => 1.15970741690448)
parametersChange2 = Dict(:k1a => "estimate", 
                        :k22 => 666.8355739795, 
                        :km2 => "estimate")
parametersChange3 = Dict(:sigmaY2Step => 5.15364156671777)
parametersChange4 = Dict(:sigmaY2Step => 5.15364156671777, 
                         :k22 => 666.8355739795)           
@testset "Test PEtab remake : BrannmarkBoehm" begin                          
    testPEtabRemake(petabModel, parametersChange1, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    testPEtabRemake(petabModel, parametersChange2, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    testPEtabRemake(petabModel, parametersChange3, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    testPEtabRemake(petabModel, parametersChange4, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
end