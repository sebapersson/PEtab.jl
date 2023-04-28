using PEtab
using Test
using OrdinaryDiffEq
# Used to test cost-value at the nominal parameter value 
function testLogLikelihoodValue(petabModel::PEtabModel, 
                                referenceValue::Float64; atol=1e-3)
    
    odeSolverOptions = getODESolverOptions(Rodas4P(), abstol=1e-8, reltol=1e-8)
    petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                gradientMethod=:ForwardDiff, 
                                hessianMethod=:ForwardDiff, 
                                splitOverConditions=true)

    cost = petabProblem.computeCost(petabProblem.θ_nominalT)
    println("Model : ", petabModel.modelName)
    @test cost ≈ referenceValue atol=atol    

end

@testset "Log likelihood values and gradients for benchmark collection" begin
    
    # Beer model - Numerically challenging gradient as we have callback rootfinding
    pathYML = joinpath(@__DIR__, "JuliaImport", "Beer", "Beer_MolBioSystems2014.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Beer", "Beer.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, -58622.9145631413)
    
    # Boehm model 
    pathYML = joinpath(@__DIR__, "JuliaImport", "Boehm", "Boehm_JProteomeRes2014.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Boehm", "Boehm.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, 138.22199693517703)
    
    # Brännmark model. Model has pre-equlibration criteria so here we test all gradients. Challenging to compute gradients.
    pathYML = joinpath(@__DIR__, "JuliaImport", "Brannmark", "Brannmark_JBC2010.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Brannmark", "Brannmark.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, 141.889113770537)
    
    # Fujita model. Challangeing to compute accurate gradients  
    pathYML = joinpath(@__DIR__, "JuliaImport", "Fujita", "Fujita_SciSignal2010.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Fujita", "Fujita.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, -53.08377736998929)
    
    # Isensee model. Accurate gradients are computed (but the code takes ages to run with low tolerances)
    pathYML = joinpath(@__DIR__, "JuliaImport", "Isensee", "Isensee_JCB2018.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Isensee", "Isensee.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, 3949.375966548649+4.45299970460275, atol=1e-2)
    
    # Weber model. Challanging as it sensitivity to steady state tolerances 
    pathYML = joinpath(@__DIR__, "JuliaImport", "Weber", "Weber_BMC2015.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Weber", "Weber.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, 296.2017922646865)

end
