# Model name: Test_model3
# Number of parameters: 6
# Number of species: 2
function getODEModel_Test_model3()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t x(t) y(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [x, y]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters c default b a_scale a d

    ### Store parameters in array for ODESystem command
    parameterArray = [c, default, b, a_scale, a, d]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(x) ~ +1.0 * ( 1 /default ) * (((a*a_scale)-(b*x))+(c*y)),
    D(y) ~ +1.0 * ( 1 /default ) * ((b*x)-((c*y)+(d*y)))
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    x => 0.0,
    y => 0.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    c => 3.0,
    default => 1.0,
    b => 2.0,
    a_scale => 1.0,
    a => 1.0,
    d => 4.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
