using PEtab
using Test
using OrdinaryDiffEq
using Printf


# Used to test cost-value at the nominal parameter value
function testLogLikelihoodValue(petabModel::PEtabModel,
                                referenceValue::Float64; atol=1e-3)

    petabProblem = createPEtabODEProblem(petabModel,
                                         odeSolverOptions=ODESolverOptions(Rodas4P(), abstol=1e-8, reltol=1e-8),
                                         gradientMethod=:ForwardDiff,
                                         hessianMethod=:ForwardDiff,
                                         splitOverConditions=true,
                                         verbose=false)

    cost = petabProblem.computeCost(petabProblem.θ_nominalT)
    strWrite = @sprintf("Model : %s", petabModel.modelName)
    @info "$strWrite"
    @test cost ≈ referenceValue atol=atol

end

@testset "Julia import and cost calculation" begin

    # Beer model - Numerically challenging gradient as we have callback rootfinding
    pathYML = joinpath(@__DIR__, "JuliaImport", "Beer", "Beer_MolBioSystems2014.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Beer", "Beer.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, -58622.9145631413)

    # Boehm model
    pathYML = joinpath(@__DIR__, "JuliaImport", "Boehm", "Boehm_JProteomeRes2014.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Boehm", "Boehm.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, 138.22199693517703)

    # Brännmark model. Model has pre-equlibration criteria so here we test all gradients. Challenging to compute gradients.
    pathYML = joinpath(@__DIR__, "JuliaImport", "Brannmark", "Brannmark_JBC2010.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Brannmark", "Brannmark.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, 141.889113770537)

    # Fujita model. Challangeing to compute accurate gradients
    pathYML = joinpath(@__DIR__, "JuliaImport", "Fujita", "Fujita_SciSignal2010.yaml")
    pathJuliaFile = joinpath(@__DIR__, "JuliaImport", "Fujita", "Fujita.jl")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFilePath=pathJuliaFile)
    testLogLikelihoodValue(petabModel, -53.08377736998929)

end
