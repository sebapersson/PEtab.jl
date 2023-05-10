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