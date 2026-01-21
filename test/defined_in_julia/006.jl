#=
    Test 0006 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS5 begin
    @parameters begin
        a0
        b0
        k1
        k2
    end
    @variables begin
        A(t) = a0
        B(t) = b0
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS5()

measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0.0, 10.0],
                         measurement=[0.7, 0.1],
                         observable_parameters=[10.0, 15.0])

simulation_conditions = [PEtabCondition(:c0)]

parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
              PEtabParameter(:b0, value=0.0, scale=:lin),
              PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

@unpack A = rn
@parameters observableParameter1_obs_a
observables = PEtabObservable(:obs_a, observableParameter1_obs_a * A, 1.0)

model_rn = PEtabModel(rn, observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_prob_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_prob_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 65.10833033589059 atol=1e-3
@test nll_sys ≈ 65.10833033589059 atol=1e-3
