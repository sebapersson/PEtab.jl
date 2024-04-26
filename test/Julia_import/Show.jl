using Printf, Distributions

# Test the structs are printed correctly
@variables t, A(t)
@test @sprintf("%s", PEtabParameter(:a0, value=1.0, scale=:lin)) == "PEtabParameter a0. Estimated on lin-scale with bounds [1.0e-03, 1.0e+03]"
@test @sprintf("%s", PEtabObservable(A, 0.5)) == "PEtabObservable: h = A(t), noise-formula = 0.5 and normal (Gaussian) measurement noise"
@test @sprintf("%s", PEtabEvent(5.0, 5, A)) == "PEtabEvent: condition t == 5.0, affect A(t) = 5"
