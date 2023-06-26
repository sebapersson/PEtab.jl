function getODEModel_0011(foo)
    ### Define independent and dependent variables
    ModelingToolkit.@variables t B(t) A(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [B, A]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters compartment k1 a0 k2 __init__B__

    ### Store parameters in array for ODESystem command
    parameterArray = [compartment, k1, a0, k2,  __init__B__]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(B) ~ +1.0 * ( 1 /compartment ) * ((compartment*k1)*A)-1.0 * ( 1 /compartment ) * ((compartment*k2)*B),
    D(A) ~ -1.0 * ( 1 /compartment ) * ((compartment*k1)*A)+1.0 * ( 1 /compartment ) * ((compartment*k2)*B)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
	B => __init__B__,
    A => a0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
	__init__B__ => 1.0,
    compartment => 1.0,
    k1 => 0.0,
    a0 => 1.0,
    k2 => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end

