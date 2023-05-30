# [Providing the model as a Julia file instead of an SBML File](@id Beer_Julia_import)

In this tutorial, we'll demonstrate how to create a `PEtabODEproblem` problem using a Julia model file instead of an SBML file.

## Create the model file

To start, we'll create a .jl file that utilizes [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) to define a system:

```julia
function BeerODEModel()

    ModelingToolkit.@variables t Glu(t) cGlu(t) Ind(t) Bac(t) lag(t)
    ModelingToolkit.@parameters kdegi medium Bacmax ksyn kdim tau init_Bac beta

    ### Store dependent variables in array for ODESystem command
    stateArray = [Glu, cGlu, Ind, Bac]
    ### Store parameters in array for ODESystem command
    parameterArray = [kdegi, medium, Bacmax, ksyn, kdim, tau, init_Bac, beta]

    ### Equations
    EquationList = [
    Differential(t)(Glu) ~ -Bac * Glu * ksyn,
    Differential(t)(cGlu) ~ Bac * Glu * ksyn - (cGlu)^(2) * kdim,
    Differential(t)(Ind) ~ cGlu^2 * kdim - Ind * kdegi,
    Differential(t)(Bac) ~ Bac * beta * lag * (Bacmax - Bac) / Bacmax,
    lag ~ ifelse(t - tau < 0, 0, 1)
    ]

    @named odeSystem = ODESystem(EquationList, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialValues = [
    Glu => 10.0,
    cGlu => 0.0,
    Ind => 0.0,
    Bac => init_Bac
    ]

    ### SBML file parameter values ###
    parameterValues = [
    kdegi => 1.0,
    Bacmax => 1.0,
    ksyn => 1.0,
    kdim => 1.0,
    tau => 1.0,
    init_Bac => 0.0147007946993721,
    beta => 1.0
    ]

    return odeSystem, initialValues, parameterValues
end

```

**NOTE** - If you are providing a model directly in Julia, there are a few things you need to keep in mind:

* The model must be defined within a function, like BeerODEModel, which returns the odeSystem, initialValues, and parameterValues in that order (the names of the variables don't matter).
* The states and parameters must be stored in arrays, like stateArray and parameterArray, and provided as arguments when creating the ODESystem. Otherwise, a parameter might get simplified away in the symbolic pre-processing.
* When creating the initial-value map, like initialValues, the initial values can be constants or a mathematical expression that depends on parameters. For example, Glu => kdegi / tau would be an acceptable option.
* If you have an event, like a dosage or a time delay (like lag), encode it using an ifelse statement. This will later be rewritten to a callback (event) In the example above we included `ifelse(t - tau < 0, 0, 1)` to capture:
```math
    lag = 
    \begin{cases}
        0 \quad t \leq  \tau \\
        1 \quad t >  \tau 
    \end{cases}
```

For more information on how to specify the model in the ModelingToolkit.jl format, please refer to their [documentation](https://github.com/SciML/ModelingToolkit.jl).

## Import the model file

To import the Julia model, we need to set the path to Julia file via the variable `jlFilePath`. By doing so, we inform `readPEtabModel` to skip searching for an SBML file and instead import the Julia model:

```julia
using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "Beer", "Beer_MolBioSystems2014.yaml") 
pathJuliaFile = joinpath(@__DIR__, "Beer", "Julia_import_files", "Beer_Julia_Import.jl")
petabModel = readPEtabModel(pathYaml, verbose=true, jlFilePath=pathJuliaFile)
```
```
PEtabModel for model Beer. ODE-system has 4 states and 9 parameters.
Generated Julia files are at ...
```

Moving forward, you can use the imported model similar to any other model imported from an SBML-file. To get an idea of how to use the wpetabModel` to compute the cost, gradient or hessian for an ODE parameter estimation problem, please refer to the tutorial for the [Beer model](@ref Beer_tut).
