#=
    Test 0007 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0 offset_A
    @species A(t)=a0 B(t)=b0
    @observables begin
        obs_a ~ A
        obs_b ~ B
    end
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS7 begin
    @parameters begin
        a0
        b0
        k1
        k2
    end
    @variables begin
        A(t) = a0
        B(t) = b0
        # Observables
        obs_a(t)
        obs_b(t)
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
        obs_a ~ A
        obs_b ~ B
    end
end
@mtkbuild sys = SYS7()

measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_b"],
                         time=[10.0, 10.0],
                         measurement=[0.2, 0.8])

simulation_conditions = [PEtabCondition(:c0)]

parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
              PEtabParameter(:b0, value=0.0, scale=:lin),
              PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

petab_observables = [PEtabObservable("obs_a", :obs_a, 0.5),
                     PEtabObservable(:obs_b, :obs_b, 0.6; distribution=PEtab.Log10Normal)]

model_rn = PEtabModel(rn, petab_observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn)
model_sys = PEtabModel(sys, petab_observables, measurements, parameters,
                       simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys)

nll_rn = petab_problem_rn.nllh(get_x(petab_problem_rn))
nll_sys = petab_problem_sys.nllh(get_x(petab_problem_sys))
@test nll_rn ≈ 1.378941036858 atol=1e-3
@test nll_sys ≈ 1.378941036858 atol=1e-3
