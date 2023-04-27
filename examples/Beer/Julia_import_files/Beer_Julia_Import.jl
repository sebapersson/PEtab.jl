# Model name: Beer
# Number of parameters: 8
# Number of species: 4
function getODEModel_Beer_Julia_Import()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t Glu(t) cGlu(t) Ind(t) Bac(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [Glu, cGlu, Ind, Bac]

    ### Define variable parameters

    ### Define potential algebraic variables
    ModelingToolkit.@variables lag(t)

    ### Define parameters
    ModelingToolkit.@parameters kdegi medium Bacmax ksyn kdim tau init_Bac beta

    ### Store parameters in array for ODESystem command
    parameterArray = [kdegi, medium, Bacmax, ksyn, kdim, tau, init_Bac, beta]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    EquationList = [
    D(Glu) ~ +1.0 * ( 1 /medium ) * (medium * -Bac * Glu * ksyn),
    D(cGlu) ~ +1.0 * ( 1 /medium ) * (medium * (Bac * Glu * ksyn - (cGlu)^(2) * kdim)),
    D(Ind) ~ +1.0 * ( 1 /medium ) * (medium * ((cGlu)^(2) * kdim - Ind * kdegi)),
    D(Bac) ~ +1.0 * ( 1 /medium ) * (medium * (Bac * beta * lag * (Bacmax + -Bac) / Bacmax)),
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
    TrueParameterValues = [
    kdegi => 1.0,
    medium => 1.0,
    Bacmax => 1.0,
    ksyn => 1.0,
    kdim => 1.0,
    tau => 1.0,
    init_Bac => 0.0147007946993721,
    beta => 1.0
    ]

    return OdeSystem, InitialSpeciesValues, TrueParameterValues

end
