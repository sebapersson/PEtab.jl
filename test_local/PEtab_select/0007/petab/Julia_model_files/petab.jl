# Model name: petab
# Number of parameters: 4
# Number of species: 2
function getODEModel_petab()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t x1(t) x2(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [x1, x2]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters default observable_x2 k3 k1 k2

    ### Store parameters in array for ODESystem command
    parameterArray = [default, observable_x2, k3, k1, k2]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(x1) ~ +1.0 * ( 1 /default ) * (default * (k1 / default))-1.0 * ( 1 /default ) * (default * (k2 * x1 / default))-1.0 * ( 1 /default ) * (default * (k3 * x1 * x2 / default)),
    D(x2) ~ +1.0 * ( 1 /default ) * (default * (k2 * x1 / default))-1.0 * ( 1 /default ) * (default * (k3 * x1 * x2 / default))
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    x1 => 0.0,
    x2 => 0.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    default => 1.0,
    observable_x2 => 0.0,
    k3 => 0.0,
    k1 => 0.2,
    k2 => 0.1
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
