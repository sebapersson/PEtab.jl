function getODEModel_Beer()
	# Model name: Beer
	# Number of parameters: 8
	# Number of species: 4

    ### Define independent and dependent variables
    ModelingToolkit.@variables t Glu(t) Ind(t) cGlu(t) Bac(t) lag(t) 

    ### Store dependent variables in array for ODESystem command
    stateArray = [Glu, Ind, cGlu, Bac]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters lag_bool1 kdegi medium Bacmax ksyn kdim tau init_Bac beta 

    ### Store parameters in array for ODESystem command
    parameterArray = [lag_bool1, kdegi, medium, Bacmax, ksyn, kdim, tau, init_Bac, beta]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(Glu) ~ -ksyn*Bac*Glu,
    D(cGlu) ~ ksyn*Bac*Glu - kdim*(cGlu^2),
    D(Ind) ~ kdim*(cGlu^2) - kdegi*Ind,
    D(Bac) ~ (beta*(Bacmax - Bac)*Bac*lag) / Bacmax,
    lag~((1 - lag_bool1)*( 0) + lag_bool1*( 1))
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
        Glu => 10.0, 
        cGlu => 0.0, 
        Ind => 0.0, 
        Bac => init_Bac
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
        kdegi => 1.0, 
        medium => 1.0, 
        Bacmax => 1.0, 
        ksyn => 1.0, 
        kdim => 1.0, 
        tau => 1.0, 
        init_Bac => 0.0147007946993721, 
        beta => 1.0, 
        lag_bool1 => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
