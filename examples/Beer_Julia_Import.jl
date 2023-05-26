#= 
    In this example we setup the PEtab problem using just a Julia file instead of an SBML-file.

    The flag jlFile=true will tell the importer to not use any SBML-files, even if they are specified in the yaml-file.
    One must though specify the path to the julia file by using the variable jlFilePath="/path/to/jl/file"

    The julia file must:
        - Contain a function specifying the model and that function must be the last function in the file.
        - Return the OdeSystem, InitialSpeciesValues, TrueParameterValues, in that specific order (the names of the variables do not matter though).

    A copy with the suffix "_fix" will be created. It will have the format needed to work with the 
    PEtab toolbox, among other things, all ifelse-statments are automatically translated into expressions 
    using boolean variables.
    
    Besides this in the example folder we also have:
    Boehm.jl - here we show how to best handle small models (states ≤ 20, parameters ≤ 20). We further cover more details 
        about the important readPEtabModel and createPEtabODEProblem functions. Recommended to checkout before looking at
        Bachmann.jl, Beer.jl and Brannmark.jl.
    Bachmann.jl - here we show how to set the best options for a medium sized model (20 ≤ states ≤ 50, 20 ≤ parameters ≤ 75), 
        and how to compute gradients via adjoint sensitivity analysis.
    Beer.jl - here we show how to handle models when majority of parameter are specific to specific experimental conditions.
    Brannmark.jl - here we show how to handle models with preequilibration (model must be simulated to steady-state)
=#

using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "Beer", "Beer_MolBioSystems2014.yaml") # @__DIR__ = file directory
pathJuliaFile = joinpath(@__DIR__, "Beer", "Julia_import_files", "Beer_Julia_Import.jl")
petabModel = readPEtabModel(pathYaml, verbose=true, jlFile=true, jlFilePath=pathJuliaFile)

#=
    The cost of the imported model is calculated. 
    For more details about the following lines, please look at the example for the Beer model that imports an SBML-model.
=#
petabProblem = createPEtabODEProblem(petabModel, 
                                     odeSolverOptions=ODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8),
                                     gradientMethod=:ForwardDiff, 
                                     hessianMethod=:ForwardDiff, 
                                     splitOverConditions=true)

p = petabProblem.θ_nominalT # Parameter values in the PEtab file on log-scale
cost = petabProblem.computeCost(p)
@printf("Cost for Beer = %.2f\n", cost)
