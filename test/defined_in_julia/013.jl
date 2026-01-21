#=
    Test 0013 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters par
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS13 begin
    @parameters begin
        k1
        k2
        par
    end
    @variables begin
        A(t) = 1.0
        B(t)
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS13()

speciemap = [:A => 1.0]

measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0.0, 10.0],
                         measurement=[0.7, 0.1])

simulation_conditions = PEtabCondition(:c0, :B => :par)

parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin),
              PEtabParameter(:par, value=7.0, scale=:lin)]

@unpack A = rn
observables = PEtabObservable(:obs_a, A, 0.5)

model_rn = PEtabModel(sys, observables, measurements, parameters; speciemap = speciemap,
                      simulation_conditions = simulation_conditions, )
petab_prob_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_prob_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 22.79033132827511 atol=1e-3
@test nll_sys ≈ 22.79033132827511 atol=1e-3
