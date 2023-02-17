# Model name: Crauste_CellSystems2017
# Number of parameters: 12
# Number of species: 5
function getODEModel_Crauste_CellSystems2017()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t Naive(t) Pathogen(t) LateEffector(t) EarlyEffector(t) Memory(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [Naive, Pathogen, LateEffector, EarlyEffector, Memory]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters mu_LL delta_NE mu_PE mu_P mu_PL delta_EL mu_EE default mu_N rho_E delta_LM rho_P mu_LE

    ### Store parameters in array for ODESystem command
    parameterArray = [mu_LL, delta_NE, mu_PE, mu_P, mu_PL, delta_EL, mu_EE, default, mu_N, rho_E, delta_LM, rho_P, mu_LE]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(Naive) ~ -1.0 * ( 1 /default ) * (Naive * mu_N)-1.0 * ( 1 /default ) * (Naive * Pathogen * delta_NE),
    D(Pathogen) ~ +1.0 * ( 1 /default ) * ((Pathogen)^(2) * rho_P)-1.0 * ( 1 /default ) * (EarlyEffector * Pathogen * mu_PE)-1.0 * ( 1 /default ) * (LateEffector * Pathogen * mu_PL)-1.0 * ( 1 /default ) * (Pathogen * mu_P),
    D(LateEffector) ~ +1.0 * ( 1 /default ) * (EarlyEffector * delta_EL)-1.0 * ( 1 /default ) * ((LateEffector)^(2) * mu_LL)-1.0 * ( 1 /default ) * (EarlyEffector * LateEffector * mu_LE)-1.0 * ( 1 /default ) * (LateEffector * delta_LM),
    D(EarlyEffector) ~ +1.0 * ( 1 /default ) * (Naive * Pathogen * delta_NE)+1.0 * ( 1 /default ) * (EarlyEffector * Pathogen * rho_E)-1.0 * ( 1 /default ) * ((EarlyEffector)^(2) * mu_EE)-1.0 * ( 1 /default ) * (EarlyEffector * delta_EL),
    D(Memory) ~ +1.0 * ( 1 /default ) * (LateEffector * delta_LM)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    Naive => 8090.0,
    Pathogen => 1.0,
    LateEffector => 0.0,
    EarlyEffector => 0.0,
    Memory => 0.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    mu_LL => 8.11520135326853e-6,
    delta_NE => 0.0119307857579241,
    mu_PE => 1.36571832778378e-10,
    mu_P => 1.00000002976846e-5,
    mu_PL => 3.6340308186265e-5,
    delta_EL => 0.51794597529254,
    mu_EE => 3.91359322673521e-5,
    default => 1.0,
    mu_N => 0.739907308603256,
    rho_E => 0.507415703707752,
    delta_LM => 0.0225806365892933,
    rho_P => 0.126382288121756,
    mu_LE => 1.00000000000005e-10
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
