# Model name: Test_model2
# Number of parameters: 3
# Number of species: 2
function getODEModel_Test_model2()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t sebastian(t) damiano(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [sebastian, damiano]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters default alpha beta

    ### Store parameters in array for ODESystem command
    parameterArray = [default, alpha, beta]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(sebastian) ~ +1.0 * ( 1 /default ) * (alpha*sebastian),
    D(damiano) ~ +1.0 * ( 1 /default ) * (beta*damiano)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    sebastian => 8.0,
    damiano => 4.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    default => 1.0,
    alpha => 5.0,
    beta => 3.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
