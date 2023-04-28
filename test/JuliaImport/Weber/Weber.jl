# Model name: Weber
# Number of parameters: 38
# Number of species: 7
function getODEModel_Weber()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t CERTERa(t) PI4K3B(t) CERT(t) CERTTGNa(t) PKDDAGa(t) PKD(t) PI4K3Ba(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [CERTERa, PI4K3B, CERT, CERTTGNa, PKDDAGa, PKD, PI4K3Ba]

    ### Define variable parameters

    ### Define potential algebraic variables
    ModelingToolkit.@variables u6(t) u4(t) u3(t) u5(t)

    ### Define parameters
    ModelingToolkit.@parameters a21 m33 kb_NB142_70_dose a22 kb_NB142_70_time p12 p33 s31 PdBu_dose a12 p22 a33 p13 cyt pu5 pu2 p31 pu4 PdBu_time s12 m11 m31 u2 Ect_Expr_PI4K3beta_flag a11 p32 p21 pu6 s21 pu3 m22 p11 Ect_Expr_CERT_flag a31 a32

    ### Store parameters in array for ODESystem command
    parameterArray = [a21, m33, kb_NB142_70_dose, a22, kb_NB142_70_time, p12, p33, s31, PdBu_dose, a12, p22, a33, p13, cyt, pu5, pu2, p31, pu4, PdBu_time, s12, m11, m31, u2, Ect_Expr_PI4K3beta_flag, a11, p32, p21, pu6, s21, pu3, m22, p11, Ect_Expr_CERT_flag, a31, a32]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    EQS = [
    D(CERTERa) ~ -1.0 * ( 1 /cyt ) * (cyt * (CERTERa * PI4K3Ba * p31 / (PI4K3Ba + m31)))+1.0 * ( 1 /cyt ) * (cyt * CERT * p32)+1.0 * ( 1 /cyt ) * (cyt * s31)+1.0 * ( 1 /cyt ) * (cyt * pu4 * u4)-1.0 * ( 1 /cyt ) * (cyt * CERTERa * a31),
    D(PI4K3B) ~ +1.0 * ( 1 /cyt ) * (cyt * PI4K3Ba * p21)-1.0 * ( 1 /cyt ) * (cyt * (PI4K3B * PKDDAGa * p22 / (PKDDAGa + m22)))+1.0 * ( 1 /cyt ) * (cyt * s21)+1.0 * ( 1 /cyt ) * (cyt * pu3 * u3)-1.0 * ( 1 /cyt ) * (cyt * PI4K3B * a21),
    D(CERT) ~ -1.0 * ( 1 /cyt ) * (cyt * CERT * p32)+1.0 * ( 1 /cyt ) * (cyt * (CERTTGNa * PKDDAGa * p33 / (PKDDAGa + m33)))-1.0 * ( 1 /cyt ) * (cyt * CERT * a32),
    D(CERTTGNa) ~ +1.0 * ( 1 /cyt ) * (cyt * (CERTERa * PI4K3Ba * p31 / (PI4K3Ba + m31)))-1.0 * ( 1 /cyt ) * (cyt * (CERTTGNa * PKDDAGa * p33 / (PKDDAGa + m33)))-1.0 * ( 1 /cyt ) * (cyt * CERTTGNa * a33),
    D(PKDDAGa) ~ +1.0 * ( 1 /cyt ) * (cyt * (CERTERa * PI4K3Ba * PKD * p11 * p31 / ((PI4K3Ba + m31) * (m11 + CERTERa * PI4K3Ba * p31 / (PI4K3Ba + m31)))))+1.0 * ( 1 /cyt ) * (cyt * PKD * p12 * (pu5 * u5 + 1))-1.0 * ( 1 /cyt ) * (cyt * PKDDAGa * p13 * (pu6 * u6 + 1))-1.0 * ( 1 /cyt ) * (cyt * PKDDAGa * a12),
    D(PKD) ~ -1.0 * ( 1 /cyt ) * (cyt * (CERTERa * PI4K3Ba * PKD * p11 * p31 / ((PI4K3Ba + m31) * (m11 + CERTERa * PI4K3Ba * p31 / (PI4K3Ba + m31)))))-1.0 * ( 1 /cyt ) * (cyt * PKD * p12 * (pu5 * u5 + 1))+1.0 * ( 1 /cyt ) * (cyt * PKDDAGa * p13 * (pu6 * u6 + 1))+1.0 * ( 1 /cyt ) * (cyt * s12)+1.0 * ( 1 /cyt ) * (cyt * pu2 * u2)-1.0 * ( 1 /cyt ) * (cyt * PKD * a11),
    D(PI4K3Ba) ~ -1.0 * ( 1 /cyt ) * (cyt * PI4K3Ba * p21)+1.0 * ( 1 /cyt ) * (cyt * (PI4K3B * PKDDAGa * p22 / (PKDDAGa + m22)))-1.0 * ( 1 /cyt ) * (cyt * PI4K3Ba * a22),
    u6 ~ kb_NB142_70_dose * ifelse(t - kb_NB142_70_time < 0, 0, 1),
    u4 ~ ifelse(t < 0, 0, Ect_Expr_CERT_flag),
    u3 ~ ifelse(t < 0, 0, Ect_Expr_PI4K3beta_flag),
    u5 ~ ifelse(t - PdBu_time < 0, 0, PdBu_dose)
    ]

    @named sys = ODESystem(EQS, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    inSpecV = [
    CERTERa => 3.19483885902e7,
    PI4K3B => 1.5775405394e6,
    CERT => 160797.7364,
    CERTTGNa => 4.20828286681e7,
    PKDDAGa => 123.8608,
    PKD => 466534.7994,
    PI4K3Ba => 332054.5041
    ]

    ### SBML file parameter values ###
    truParVal = [
    a21 => 1.86921330588484,
    m33 => 7.56019670948633e7,
    kb_NB142_70_dose => 0.0,
    a22 => 0.00010000000000001,
    kb_NB142_70_time => 0.0,
    p12 => 0.00601235513713934,
    p33 => 40184.0793611631,
    s31 => 4.34566094708907e6,
    PdBu_dose => 0.0,
    a12 => 25.9422257786565,
    p22 => 21.3989931197985,
    a33 => 0.00010000000000001,
    p13 => 0.00183182922741504,
    cyt => 1.0,
    pu5 => 33.7869036338953,
    pu2 => 1.0,
    p31 => 2428.01870197136,
    pu4 => 3.57637576834332e7,
    PdBu_time => 0.0,
    s12 => 88884.6918603076,
    m11 => 9.99999999999902e9,
    m31 => 9.98783985509162e9,
    u2 => 0.0,
    Ect_Expr_PI4K3beta_flag => 0.0,
    a11 => 0.183516872816456,
    p32 => 17.1128427194665,
    p21 => 5.73048563386363,
    pu6 => 147.540504730735,
    s21 => 2.95440507105195e6,
    pu3 => 9.99999999999949e7,
    m22 => 2174.18367530733,
    p11 => 4.23578569731751,
    Ect_Expr_CERT_flag => 0.0,
    a31 => 0.140427319109888,
    a32 => 0.00010000000000001
    ]

    return sys, inSpecV, truParVal

end
