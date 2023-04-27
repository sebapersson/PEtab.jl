#= 
    In this example we setup the PEtab problem using just a Julia file instead of an SBML-file.

    The flag jlFile=true will tell the importer to not use any SBML-files, even if they are specified in the yaml-file.
    The julia file must:
        - Be located inside a folder called Julia_model_files
        - Have a function named getODEModel_M where M is the model name
        - The file must be named M.jl where M is the model name (the same as in getODEModel_M)
        - Return the OdeSystem, InitialSpeciesValues, TrueParameterValues, in that order (the names of the variables do not matter though).

    A file named M_fix.jl, where M is the model name, is created in the folder Julia_model_files. It will have the format needed to work with the 
    PEtab toolbox, among other things, all ifelse-statments are automatically translated into expressions 
    using boolean variables.
    
    Besides this in the example folder we also have:
    Boehm.jl - here we show how to best handle small models (states ≤ 20, parameters ≤ 20). We further cover more details 
        about the important readPEtabModel and setupPEtabODEProblem functions. Recommended to checkout before looking at
        Bachmann.jl, Beer.jl and Brannmark.jl.
    Bachmann.jl - here we show how to set the best options for a medium sized model (20 ≤ states ≤ 50, 20 ≤ parameters ≤ 75), 
        and how to compute gradients via adjoint sensitivity analysis.
    Beer.jl - here we show how to handle models when majority of parameter are specific to specific experimental conditions.
    Brannmark.jl - here we show how to handle models with preequilibration (model must be simulated to steady-state)
=#

using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "Beer_Julia_Import", "Beer_MolBioSystems2014.yaml") # @__DIR__ = file directory
petabModel = readPEtabModel(pathYaml, verbose=true, jlFile=true)

#=
    The cost of the imported model is calculated. 
    For more details about the following lines, please look at the example for the Beer model that imports an SBML-model.
=#
odeSolverOptions = getODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:ForwardDiff, 
                                    splitOverConditions=true)

p = petabProblem.θ_nominalT # Parameter values in the PEtab file on log-scale
cost = petabProblem.computeCost(p)
@printf("Cost for Beer = %.2f\n", cost)