function getODEModel_Zheng_PNAS2012(foo)
	# Model name: Zheng_PNAS2012
	# Number of parameters: 47
	# Number of species: 15

    ### Define independent and dependent variables
    ModelingToolkit.@variables t K27me0K36me0(t) K27me2K36me3(t) K27me2K36me0(t) K27me0K36me1(t) K27me2K36me1(t) K27me0K36me2(t) K27me1K36me2(t) K27me3K36me2(t) K27me1K36me1(t) K27me1K36me3(t) K27me2K36me2(t) K27me0K36me3(t) K27me3K36me1(t) K27me3K36me0(t) K27me1K36me0(t) inflow(t) 

    ### Store dependent variables in array for ODESystem command
    stateArray = [K27me0K36me0, K27me2K36me3, K27me2K36me0, K27me0K36me1, K27me2K36me1, K27me0K36me2, K27me1K36me2, K27me3K36me2, K27me1K36me1, K27me1K36me3, K27me2K36me2, K27me0K36me3, K27me3K36me1, K27me3K36me0, K27me1K36me0, inflow]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters k20_10 k11_12 k22_32 k21_11 k13_12 k01_00 k11_01 k22_21 k22_12 k11_10 k12_11 k21_31 k02_12 k30_20 dilution k31_21 k23_22 k02_01 k03_13 k10_11 k21_20 k10_00 k30_31 k20_30 k20_21 k00_10 k12_13 k01_11 k02_03 k00_01 k03_02 k23_13 k32_31 default k13_03 k31_30 k12_02 k31_32 k01_02 k13_23 k21_22 k12_22 k11_21 k22_23 k10_20 inflowp k32_22 

    ### Store parameters in array for ODESystem command
    parameterArray = [k20_10, k11_12, k22_32, k21_11, k13_12, k01_00, k11_01, k22_21, k22_12, k11_10, k12_11, k21_31, k02_12, k30_20, dilution, k31_21, k23_22, k02_01, k03_13, k10_11, k21_20, k10_00, k30_31, k20_30, k20_21, k00_10, k12_13, k01_11, k02_03, k00_01, k03_02, k23_13, k32_31, default, k13_03, k31_30, k12_02, k31_32, k01_02, k13_23, k21_22, k12_22, k11_21, k22_23, k10_20, inflowp, k32_22]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Derivatives ###
    eqs = [
    D(K27me0K36me0) ~ +1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(k01_00))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me0)*(k00_10))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me0)*(k00_01))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(k10_00))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me0)*(inflow))/(default)))+1.0 * ( 1 /default ) * (default*((inflow)/(default))),
    D(K27me2K36me3) ~ -1.0 * ( 1 /default ) * (default*(((K27me2K36me3)*(k23_22))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(k13_23))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_23))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me3)*(k23_13))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me3)*(inflow))/(default))),
    D(K27me2K36me0) ~ -1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(k20_30))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(inflow))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me3K36me0)*(k30_20))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_20))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(k10_20))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(k20_10))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(k20_21))/(default))),
    D(K27me0K36me1) ~ -1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(k01_00))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(k01_02))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(k02_01))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_01))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(k01_11))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me0)*(k00_01))/(default))),
    D(K27me2K36me1) ~ -1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_20))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_22))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(k20_21))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_21))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_31))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_11))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(k31_21))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_21))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(inflow))/(default))),
    D(K27me0K36me2) ~ +1.0 * ( 1 /default ) * (default*(((K27me0K36me3)*(k03_02))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(k02_03))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(k01_02))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(k02_12))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(k02_01))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(inflow))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_02))/(default))),
    D(K27me1K36me2) ~ -1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_11))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_22))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_13))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(k13_12))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(k02_12))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_02))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(inflow))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_12))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_12))/(default))),
    D(K27me3K36me2) ~ -1.0 * ( 1 /default ) * (default*(((K27me3K36me2)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me2)*(k32_31))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me2)*(k32_22))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(k31_32))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_32))/(default))),
    D(K27me1K36me1) ~ +1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_11))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(k10_11))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_01))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me1)*(k01_11))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_10))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_12))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_11))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_21))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(inflow))/(default))),
    D(K27me1K36me3) ~ -1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(k13_23))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(inflow))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_13))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(k13_12))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(k13_03))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me3)*(k23_13))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me3)*(k03_13))/(default))),
    D(K27me2K36me2) ~ +1.0 * ( 1 /default ) * (default*(((K27me1K36me2)*(k12_22))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me3)*(k23_22))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_23))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_22))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me3K36me2)*(k32_22))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_21))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_32))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me2K36me2)*(k22_12))/(default))),
    D(K27me0K36me3) ~ -1.0 * ( 1 /default ) * (default*(((K27me0K36me3)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me3)*(k03_02))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me2)*(k02_03))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me3)*(k13_03))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me0K36me3)*(k03_13))/(default))),
    D(K27me3K36me1) ~ +1.0 * ( 1 /default ) * (default*(((K27me3K36me2)*(k32_31))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(k31_32))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me3K36me0)*(k30_31))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me1)*(k21_31))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(k31_21))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(k31_30))/(default))),
    D(K27me3K36me0) ~ +1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(k20_30))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me0)*(k30_20))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me0)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me3K36me0)*(k30_31))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me3K36me1)*(k31_30))/(default))),
    D(K27me1K36me0) ~ -1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(inflow))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(k10_11))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me0K36me0)*(k00_10))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(k10_20))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me2K36me0)*(k20_10))/(default)))+1.0 * ( 1 /default ) * (default*(((K27me1K36me1)*(k11_10))/(default)))-1.0 * ( 1 /default ) * (default*(((K27me1K36me0)*(k10_00))/(default))),
    inflow ~ inflowp*dilution
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    K27me0K36me0 => 0.00417724976345759,
    K27me2K36me3 => 0.00471831436002134,
    K27me2K36me0 => 0.00632744816295157,
    K27me0K36me1 => 0.0102104668587641,
    K27me2K36me1 => 0.0143896310177379,
    K27me0K36me2 => 0.169690316239546,
    K27me1K36me2 => 0.594249755169037,
    K27me3K36me2 => 0.00136041631795562,
    K27me1K36me1 => 0.0078328187288069,
    K27me1K36me3 => 0.102748675077958,
    K27me2K36me2 => 0.0263372634996529,
    K27me0K36me3 => 0.0504935214807544,
    K27me3K36me1 => 0.00250831034920277,
    K27me3K36me0 => 0.00330168411604165,
    K27me1K36me0 => 0.00165412810279407,
    inflow => inflowp*dilution
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    k20_10 => 1.00000000000008e-5,
    k11_12 => 7.20477305381394,
    k22_32 => 1.00000000004307e-5,
    k21_11 => 1.00000000000008e-5,
    k13_12 => 0.103481305789201,
    k01_00 => 1.00000000000008e-5,
    k11_01 => 1.00000000000008e-5,
    k22_21 => 1.00000000000005e-5,
    k22_12 => 0.0693602284002711,
    k11_10 => 208.897614797522,
    k12_11 => 0.0697010300045283,
    k21_31 => 1.00000000000008e-5,
    k02_12 => 0.0258535237204994,
    k30_20 => 1.00000000000168e-5,
    dilution => 0.0,
    k31_21 => 1.00000000000008e-5,
    k23_22 => 1.00000000000008e-5,
    k02_01 => 0.0365191563528239,
    k03_13 => 0.0269708538387512,
    k10_11 => 999.999999501161,
    k21_20 => 0.150813306514596,
    k10_00 => 1.00000000000008e-5,
    k30_31 => 0.567094806715041,
    k20_30 => 0.312045380417727,
    k20_21 => 1.00000000000077e-5,
    k00_10 => 4.29039735572565,
    k12_13 => 0.0189710284242353,
    k01_11 => 1.00000000000008e-5,
    k02_03 => 0.0172219162989543,
    k00_01 => 3.07977512445142,
    k03_02 => 1.00000000000008e-5,
    k23_13 => 0.249204587936977,
    k32_31 => 1.00000009388004e-5,
    default => 1.0,
    k13_03 => 1.00000000000723e-5,
    k31_30 => 1.00000000152072e-5,
    k12_02 => 1.00000000000008e-5,
    k31_32 => 0.71559252428414,
    k01_02 => 1.83597321270819,
    k13_23 => 1.00000000000008e-5,
    k21_22 => 1.00000000000008e-5,
    k12_22 => 0.00371718625668441,
    k11_21 => 0.333864323977212,
    k22_23 => 0.0501463279005419,
    k10_20 => 1.0000000000005e-5,
    inflowp => 0.0309160767779193,
    k32_22 => 1.28866373067424
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
