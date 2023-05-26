# Model name: petab
# Number of parameters: 34
# Number of species: 16
function getODEModel_petab()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t x_k05k16(t) x_k05k08k16(t) x_k05k12k16(t) x_k16(t) x_0ac(t) x_k12(t) x_k12k16(t) x_k05k12(t) x_k08(t) x_k05k08(t) x_k08k16(t) x_k08k12(t) x_k05k08k12(t) x_k05(t) x_4ac(t) x_k08k12k16(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [x_k05k16, x_k05k08k16, x_k05k12k16, x_k16, x_0ac, x_k12, x_k12k16, x_k05k12, x_k08, x_k05k08, x_k08k16, x_k08k12, x_k05k08k12, x_k05, x_4ac, x_k08k12k16]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters a_k05_k05k12 a_0ac_k16 a_k08_k08k12 a_k16_k08k16 a_k05k08k16_4ac a_k05k16_k05k08k16 a_k05k16_k05k12k16 a_k05k08_k05k08k16 a_k08k12_k08k12k16 a_k12k16_k05k12k16 a_k12_k12k16 a_0ac_k12 a_b a_k05_k05k08 a_k08k16_k05k08k16 a_k08k12k16_4ac a_k05k08k12_4ac a_k05k08_k05k08k12 a_k08k16_k08k12k16 a_k05k12k16_4ac a_k05k12_k05k12k16 a_k08_k05k08 a_k08_k08k16 a_k05k12_k05k08k12 a_k12k16_k08k12k16 a_k12_k05k12 da_b compartment a_0ac_k08 a_0ac_k05 a_k16_k05k16 a_k08k12_k05k08k12 a_k05_k05k16 a_k12_k08k12 a_k16_k12k16

    ### Store parameters in array for ODESystem command
    parameterArray = [a_k05_k05k12, a_0ac_k16, a_k08_k08k12, a_k16_k08k16, a_k05k08k16_4ac, a_k05k16_k05k08k16, a_k05k16_k05k12k16, a_k05k08_k05k08k16, a_k08k12_k08k12k16, a_k12k16_k05k12k16, a_k12_k12k16, a_0ac_k12, a_b, a_k05_k05k08, a_k08k16_k05k08k16, a_k08k12k16_4ac, a_k05k08k12_4ac, a_k05k08_k05k08k12, a_k08k16_k08k12k16, a_k05k12k16_4ac, a_k05k12_k05k12k16, a_k08_k05k08, a_k08_k08k16, a_k05k12_k05k08k12, a_k12k16_k08k12k16, a_k12_k05k12, da_b, compartment, a_0ac_k08, a_0ac_k05, a_k16_k05k16, a_k08k12_k05k08k12, a_k05_k05k16, a_k12_k08k12, a_k16_k12k16]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(x_k05k16) ~ +1.0 * ( 1 /compartment ) * (a_k05_k05k16 * a_b * x_k05)+1.0 * ( 1 /compartment ) * (a_k16_k05k16 * a_b * x_k16)-1.0 * ( 1 /compartment ) * (a_k05k16_k05k08k16 * a_b * x_k05k16)-1.0 * ( 1 /compartment ) * (a_k05k16_k05k12k16 * a_b * x_k05k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k16)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08k16)+1.0 * ( 1 /compartment ) * (da_b * x_k05k12k16),
    D(x_k05k08k16) ~ +1.0 * ( 1 /compartment ) * (a_k05k08_k05k08k16 * a_b * x_k05k08)+1.0 * ( 1 /compartment ) * (a_k05k16_k05k08k16 * a_b * x_k05k16)+1.0 * ( 1 /compartment ) * (a_k08k16_k05k08k16 * a_b * x_k08k16)-1.0 * ( 1 /compartment ) * (a_k05k08k16_4ac * a_b * x_k05k08k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08k16)+1.0 * ( 1 /compartment ) * (da_b * x_4ac),
    D(x_k05k12k16) ~ +1.0 * ( 1 /compartment ) * (a_k05k12_k05k12k16 * a_b * x_k05k12)+1.0 * ( 1 /compartment ) * (a_k05k16_k05k12k16 * a_b * x_k05k16)+1.0 * ( 1 /compartment ) * (a_k12k16_k05k12k16 * a_b * x_k12k16)-1.0 * ( 1 /compartment ) * (a_k05k12k16_4ac * a_b * x_k05k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k05k12k16)+1.0 * ( 1 /compartment ) * (da_b * x_4ac),
    D(x_k16) ~ +1.0 * ( 1 /compartment ) * (a_0ac_k16 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_k16_k05k16 * a_b * x_k16)-1.0 * ( 1 /compartment ) * (a_k16_k08k16 * a_b * x_k16)-1.0 * ( 1 /compartment ) * (a_k16_k12k16 * a_b * x_k16)-1.0 * ( 1 /compartment ) * (da_b * x_k16)+1.0 * ( 1 /compartment ) * (da_b * x_k05k16)+1.0 * ( 1 /compartment ) * (da_b * x_k08k16)+1.0 * ( 1 /compartment ) * (da_b * x_k12k16),
    D(x_0ac) ~ -1.0 * ( 1 /compartment ) * (a_0ac_k05 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_0ac_k08 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_0ac_k12 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_0ac_k16 * a_b * x_0ac)+1.0 * ( 1 /compartment ) * (da_b * x_k05)+1.0 * ( 1 /compartment ) * (da_b * x_k08)+1.0 * ( 1 /compartment ) * (da_b * x_k12)+1.0 * ( 1 /compartment ) * (da_b * x_k16),
    D(x_k12) ~ +1.0 * ( 1 /compartment ) * (a_0ac_k12 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_k12_k05k12 * a_b * x_k12)-1.0 * ( 1 /compartment ) * (a_k12_k08k12 * a_b * x_k12)-1.0 * ( 1 /compartment ) * (a_k12_k12k16 * a_b * x_k12)-1.0 * ( 1 /compartment ) * (da_b * x_k12)+1.0 * ( 1 /compartment ) * (da_b * x_k05k12)+1.0 * ( 1 /compartment ) * (da_b * x_k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_k12k16),
    D(x_k12k16) ~ +1.0 * ( 1 /compartment ) * (a_k12_k12k16 * a_b * x_k12)+1.0 * ( 1 /compartment ) * (a_k16_k12k16 * a_b * x_k16)-1.0 * ( 1 /compartment ) * (a_k12k16_k05k12k16 * a_b * x_k12k16)-1.0 * ( 1 /compartment ) * (a_k12k16_k08k12k16 * a_b * x_k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k12k16)+1.0 * ( 1 /compartment ) * (da_b * x_k05k12k16)+1.0 * ( 1 /compartment ) * (da_b * x_k08k12k16),
    D(x_k05k12) ~ +1.0 * ( 1 /compartment ) * (a_k05_k05k12 * a_b * x_k05)+1.0 * ( 1 /compartment ) * (a_k12_k05k12 * a_b * x_k12)-1.0 * ( 1 /compartment ) * (a_k05k12_k05k08k12 * a_b * x_k05k12)-1.0 * ( 1 /compartment ) * (a_k05k12_k05k12k16 * a_b * x_k05k12)-1.0 * ( 1 /compartment ) * (da_b * x_k05k12)-1.0 * ( 1 /compartment ) * (da_b * x_k05k12)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_k05k12k16),
    D(x_k08) ~ +1.0 * ( 1 /compartment ) * (a_0ac_k08 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_k08_k05k08 * a_b * x_k08)-1.0 * ( 1 /compartment ) * (a_k08_k08k12 * a_b * x_k08)-1.0 * ( 1 /compartment ) * (a_k08_k08k16 * a_b * x_k08)-1.0 * ( 1 /compartment ) * (da_b * x_k08)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08)+1.0 * ( 1 /compartment ) * (da_b * x_k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_k08k16),
    D(x_k05k08) ~ +1.0 * ( 1 /compartment ) * (a_k05_k05k08 * a_b * x_k05)+1.0 * ( 1 /compartment ) * (a_k08_k05k08 * a_b * x_k08)-1.0 * ( 1 /compartment ) * (a_k05k08_k05k08k12 * a_b * x_k05k08)-1.0 * ( 1 /compartment ) * (a_k05k08_k05k08k16 * a_b * x_k05k08)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08k16),
    D(x_k08k16) ~ +1.0 * ( 1 /compartment ) * (a_k08_k08k16 * a_b * x_k08)+1.0 * ( 1 /compartment ) * (a_k16_k08k16 * a_b * x_k16)-1.0 * ( 1 /compartment ) * (a_k08k16_k05k08k16 * a_b * x_k08k16)-1.0 * ( 1 /compartment ) * (a_k08k16_k08k12k16 * a_b * x_k08k16)-1.0 * ( 1 /compartment ) * (da_b * x_k08k16)-1.0 * ( 1 /compartment ) * (da_b * x_k08k16)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08k16)+1.0 * ( 1 /compartment ) * (da_b * x_k08k12k16),
    D(x_k08k12) ~ +1.0 * ( 1 /compartment ) * (a_k08_k08k12 * a_b * x_k08)+1.0 * ( 1 /compartment ) * (a_k12_k08k12 * a_b * x_k12)-1.0 * ( 1 /compartment ) * (a_k08k12_k05k08k12 * a_b * x_k08k12)-1.0 * ( 1 /compartment ) * (a_k08k12_k08k12k16 * a_b * x_k08k12)-1.0 * ( 1 /compartment ) * (da_b * x_k08k12)-1.0 * ( 1 /compartment ) * (da_b * x_k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_k08k12k16),
    D(x_k05k08k12) ~ +1.0 * ( 1 /compartment ) * (a_k05k08_k05k08k12 * a_b * x_k05k08)+1.0 * ( 1 /compartment ) * (a_k05k12_k05k08k12 * a_b * x_k05k12)+1.0 * ( 1 /compartment ) * (a_k08k12_k05k08k12 * a_b * x_k08k12)-1.0 * ( 1 /compartment ) * (a_k05k08k12_4ac * a_b * x_k05k08k12)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08k12)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08k12)-1.0 * ( 1 /compartment ) * (da_b * x_k05k08k12)+1.0 * ( 1 /compartment ) * (da_b * x_4ac),
    D(x_k05) ~ +1.0 * ( 1 /compartment ) * (a_0ac_k05 * a_b * x_0ac)-1.0 * ( 1 /compartment ) * (a_k05_k05k08 * a_b * x_k05)-1.0 * ( 1 /compartment ) * (a_k05_k05k12 * a_b * x_k05)-1.0 * ( 1 /compartment ) * (a_k05_k05k16 * a_b * x_k05)-1.0 * ( 1 /compartment ) * (da_b * x_k05)+1.0 * ( 1 /compartment ) * (da_b * x_k05k08)+1.0 * ( 1 /compartment ) * (da_b * x_k05k12)+1.0 * ( 1 /compartment ) * (da_b * x_k05k16),
    D(x_4ac) ~ +1.0 * ( 1 /compartment ) * (a_k05k08k12_4ac * a_b * x_k05k08k12)+1.0 * ( 1 /compartment ) * (a_k05k08k16_4ac * a_b * x_k05k08k16)+1.0 * ( 1 /compartment ) * (a_k05k12k16_4ac * a_b * x_k05k12k16)+1.0 * ( 1 /compartment ) * (a_k08k12k16_4ac * a_b * x_k08k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_4ac)-1.0 * ( 1 /compartment ) * (da_b * x_4ac)-1.0 * ( 1 /compartment ) * (da_b * x_4ac)-1.0 * ( 1 /compartment ) * (da_b * x_4ac),
    D(x_k08k12k16) ~ +1.0 * ( 1 /compartment ) * (a_k08k12_k08k12k16 * a_b * x_k08k12)+1.0 * ( 1 /compartment ) * (a_k08k16_k08k12k16 * a_b * x_k08k16)+1.0 * ( 1 /compartment ) * (a_k12k16_k08k12k16 * a_b * x_k12k16)-1.0 * ( 1 /compartment ) * (a_k08k12k16_4ac * a_b * x_k08k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k08k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k08k12k16)-1.0 * ( 1 /compartment ) * (da_b * x_k08k12k16)+1.0 * ( 1 /compartment ) * (da_b * x_4ac)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    x_k05k16 => 0.0,
    x_k05k08k16 => 0.0,
    x_k05k12k16 => 0.0,
    x_k16 => 0.0,
    x_0ac => 1.0,
    x_k12 => 0.0,
    x_k12k16 => 0.0,
    x_k05k12 => 0.0,
    x_k08 => 0.0,
    x_k05k08 => 0.0,
    x_k08k16 => 0.0,
    x_k08k12 => 0.0,
    x_k05k08k12 => 0.0,
    x_k05 => 0.0,
    x_4ac => 0.0,
    x_k08k12k16 => 0.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    a_k05_k05k12 => 0.0,
    a_0ac_k16 => 0.0,
    a_k08_k08k12 => 0.0,
    a_k16_k08k16 => 0.0,
    a_k05k08k16_4ac => 0.0,
    a_k05k16_k05k08k16 => 0.0,
    a_k05k16_k05k12k16 => 0.0,
    a_k05k08_k05k08k16 => 0.0,
    a_k08k12_k08k12k16 => 0.0,
    a_k12k16_k05k12k16 => 0.0,
    a_k12_k12k16 => 0.0,
    a_0ac_k12 => 0.0,
    a_b => 0.0,
    a_k05_k05k08 => 0.0,
    a_k08k16_k05k08k16 => 0.0,
    a_k08k12k16_4ac => 0.0,
    a_k05k08k12_4ac => 0.0,
    a_k05k08_k05k08k12 => 0.0,
    a_k08k16_k08k12k16 => 0.0,
    a_k05k12k16_4ac => 0.0,
    a_k05k12_k05k12k16 => 0.0,
    a_k08_k05k08 => 0.0,
    a_k08_k08k16 => 0.0,
    a_k05k12_k05k08k12 => 0.0,
    a_k12k16_k08k12k16 => 0.0,
    a_k12_k05k12 => 0.0,
    da_b => 1.0,
    compartment => 1.0,
    a_0ac_k08 => 0.0,
    a_0ac_k05 => 0.0,
    a_k16_k05k16 => 0.0,
    a_k08k12_k05k08k12 => 0.0,
    a_k05_k05k16 => 0.0,
    a_k12_k08k12 => 0.0,
    a_k16_k12k16 => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
