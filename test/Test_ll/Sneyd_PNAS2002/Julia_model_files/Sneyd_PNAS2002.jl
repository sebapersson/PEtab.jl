# Model name: Sneyd_PNAS2002
# Number of parameters: 19
# Number of species: 6
function getODEModel_Sneyd_PNAS2002()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t IPR_S(t) IPR_I2(t) IPR_R(t) IPR_O(t) IPR_I1(t) IPR_A(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [IPR_S, IPR_I2, IPR_R, IPR_O, IPR_I1, IPR_A]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters l_4 k_4 IP3 k4 k_2 l2 l_2 default l_6 k1 k_3 l6 membrane k3 l4 Ca k2 k_1

    ### Store parameters in array for ODESystem command
    parameterArray = [l_4, k_4, IP3, k4, k_2, l2, l_2, default, l_6, k1, k_3, l6, membrane, k3, l4, Ca, k2, k_1]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(IPR_S) ~ +1.0 * ( 1 /membrane ) * (k3 * (k_4 / k4 / (l_6 / l6)) / ((k_4 / k4 / (l_6 / l6)) + Ca) * IPR_O)-1.0 * ( 1 /membrane ) * (k_3 * IPR_S),
    D(IPR_I2) ~ +1.0 * ( 1 /membrane ) * ((k1 * (k_1 / k1 / (l_2 / l2)) + l2) * Ca / ((k_1 / k1 / (l_2 / l2)) + Ca) * IPR_A)-1.0 * ( 1 /membrane ) * ((k_1 + l_2) * IPR_I2),
    D(IPR_R) ~ +1.0 * ( 1 /membrane ) * ((k_2 + l_4 * Ca) / (1 + Ca / (k_4 / k4 / (l_6 / l6))) * IPR_O)-1.0 * ( 1 /membrane ) * ((k2 * (k_2 / k2 / (l_4 / l4)) + l4 * Ca) / ((k_2 / k2 / (l_4 / l4)) + Ca * (1 + (k_2 / k2 / (l_4 / l4)) / (k_1 / k1 / (l_2 / l2)))) * IP3 * IPR_R)-1.0 * ( 1 /membrane ) * ((k1 * (k_1 / k1 / (l_2 / l2)) + l2) * Ca / ((k_1 / k1 / (l_2 / l2)) + Ca * (1 + (k_1 / k1 / (l_2 / l2)) / (k_2 / k2 / (l_4 / l4)))) * IPR_R)+1.0 * ( 1 /membrane ) * ((k_1 + l_2) * IPR_I1),
    D(IPR_O) ~ -1.0 * ( 1 /membrane ) * ((k_2 + l_4 * Ca) / (1 + Ca / (k_4 / k4 / (l_6 / l6))) * IPR_O)+1.0 * ( 1 /membrane ) * ((k2 * (k_2 / k2 / (l_4 / l4)) + l4 * Ca) / ((k_2 / k2 / (l_4 / l4)) + Ca * (1 + (k_2 / k2 / (l_4 / l4)) / (k_1 / k1 / (l_2 / l2)))) * IP3 * IPR_R)-1.0 * ( 1 /membrane ) * ((k4 * (k_4 / k4 / (l_6 / l6)) + l6) * Ca / ((k_4 / k4 / (l_6 / l6)) + Ca) * IPR_O)+1.0 * ( 1 /membrane ) * ((k_1 / k1 / (l_2 / l2)) * (k_4 + l_6) / ((k_1 / k1 / (l_2 / l2)) + Ca) * IPR_A)-1.0 * ( 1 /membrane ) * (k3 * (k_4 / k4 / (l_6 / l6)) / ((k_4 / k4 / (l_6 / l6)) + Ca) * IPR_O)+1.0 * ( 1 /membrane ) * (k_3 * IPR_S),
    D(IPR_I1) ~ +1.0 * ( 1 /membrane ) * ((k1 * (k_1 / k1 / (l_2 / l2)) + l2) * Ca / ((k_1 / k1 / (l_2 / l2)) + Ca * (1 + (k_1 / k1 / (l_2 / l2)) / (k_2 / k2 / (l_4 / l4)))) * IPR_R)-1.0 * ( 1 /membrane ) * ((k_1 + l_2) * IPR_I1),
    D(IPR_A) ~ +1.0 * ( 1 /membrane ) * ((k4 * (k_4 / k4 / (l_6 / l6)) + l6) * Ca / ((k_4 / k4 / (l_6 / l6)) + Ca) * IPR_O)-1.0 * ( 1 /membrane ) * ((k_1 / k1 / (l_2 / l2)) * (k_4 + l_6) / ((k_1 / k1 / (l_2 / l2)) + Ca) * IPR_A)-1.0 * ( 1 /membrane ) * ((k1 * (k_1 / k1 / (l_2 / l2)) + l2) * Ca / ((k_1 / k1 / (l_2 / l2)) + Ca) * IPR_A)+1.0 * ( 1 /membrane ) * ((k_1 + l_2) * IPR_I2)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    IPR_S => 0.0,
    IPR_I2 => 0.0,
    IPR_R => 1.0,
    IPR_O => 0.0,
    IPR_I1 => 0.0,
    IPR_A => 0.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    l_4 => 0.0138802036171304,
    k_4 => 3079.207324879,
    IP3 => 0.0,
    k4 => 99938.2576283137,
    k_2 => 0.00100249472532433,
    l2 => 0.940077018858088,
    l_2 => 0.347664459128102,
    default => 1.0,
    l_6 => 0.00100000000000008,
    k1 => 3.72721095730996,
    k_3 => 1.91463005974811,
    l6 => 99999.9999999914,
    membrane => 1.0,
    k3 => 15.7453406923705,
    l4 => 2.85837713545253,
    Ca => 10.0,
    k2 => 99999.9999999914,
    k_1 => 0.923924728172175
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
