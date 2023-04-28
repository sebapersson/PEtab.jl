# [Import Julia file instead of SBML-file](@id Beer_Julia_import)

In this tutorial we will setup a `PEtabODEproblem` problem using just a Julia file instead of an SBML-file.

## Create the model file

First we create a `.jl` file that uses [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) to specify a system:

```julia
function BeerOdeModel()

    ModelingToolkit.@variables t Glu(t) cGlu(t) Ind(t) Bac(t) lag(t)
    ModelingToolkit.@parameters kdegi medium Bacmax ksyn kdim tau init_Bac beta

    ### Store dependent variables in array for ODESystem command
    stateArray = [Glu, cGlu, Ind, Bac]
    ### Store parameters in array for ODESystem command
    parameterArray = [kdegi, medium, Bacmax, ksyn, kdim, tau, init_Bac, beta]

    ### Equations
    EquationList = [
    Differential(t)(Glu) ~ +1.0 * ( 1 /medium ) * (medium * -Bac * Glu * ksyn),
    Differential(t)(cGlu) ~ +1.0 * ( 1 /medium ) * (medium * (Bac * Glu * ksyn - (cGlu)^(2) * kdim)),
    Differential(t)(Ind) ~ +1.0 * ( 1 /medium ) * (medium * ((cGlu)^(2) * kdim - Ind * kdegi)),
    Differential(t)(Bac) ~ +1.0 * ( 1 /medium ) * (medium * (Bac * beta * lag * (Bacmax + -Bac) / Bacmax)),
    lag ~ ifelse(t - tau < 0, 0, 1)
    ]

    @named OdeSystem = ODESystem(EquationList, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    InitialSpeciesValues = [
    Glu => 10.0,
    cGlu => 0.0,
    Ind => 0.0,
    Bac => init_Bac
    ]

    ### SBML file parameter values ###
    paramValues = [
    kdegi => 1.0,
    medium => 1.0,
    Bacmax => 1.0,
    ksyn => 1.0,
    kdim => 1.0,
    tau => 1.0,
    init_Bac => 0.0147007946993721,
    beta => 1.0
    ]

    return OdeSystem, InitialSpeciesValues, paramValues

end
```

Please note that the file must return the system, the initial species concentrations, and the parameter values, in that exact order. The names of the variables do not matter, though.

## Import the model file

To import the Julia model we need to specify that the `readPEtabModel` should skip looking for an SBML-file and just import the Julia model. We do this by setting the flag `jlFile` to `true` and by specifying the path to the Julia file using the variable `jlFilePath`, like so:

```julia
using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "examples", "Beer", "Beer_MolBioSystems2014.yaml") # @__DIR__ = file directory
pathJuliaFile = joinpath(@__DIR__, "examples", "Beer", "Julia_import_files", "Beer_Julia_Import.jl")
petabModel = readPEtabModel(pathYaml, verbose=true, jlFile=true, jlFilePath=pathJuliaFile)
```

From now on the model can be used in the same manner as any model imported from an SBML-file. Please see the tutorial for the [Beer model](@ref Beer_tut) to see how the `petabModel` can be used to calculate the cost, gradient or hessian for an ODE parameter estimation problem.
