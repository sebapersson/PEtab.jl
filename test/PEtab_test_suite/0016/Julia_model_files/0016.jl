# Model name: 0016
# Number of parameters: 5
# Number of species: 2
function getODEModel_0016()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t B(t) A(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [B, A]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters compartment b0 k1 a0 k2

    ### Store parameters in array for ODESystem command
    parameterArray = [compartment, b0, k1, a0, k2]

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
    B => b0,
    A => a0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    compartment => 1.0,
    b0 => 1.0,
    k1 => 0.0,
    a0 => 1.0,
    k2 => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
