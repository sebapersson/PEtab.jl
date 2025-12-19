#=
    Test 0009 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS9 begin
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
@mtkbuild sys = SYS9()

measurements = DataFrame(pre_eq_id=["preeq_c0", "preeq_c0"],
                         simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[1.0, 10.0],
                         measurement=[0.7, 0.1])

simulation_conditions = [PEtabCondition(:c0, :k1 => 0.8)
                         PEtabCondition(:preeq_c0, :k1 => 0.3)]

parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
              PEtabParameter(:b0, value=0.0, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

@unpack A = rn
observables = PEtabObservable("obs_a", A, 0.5)

model_rn = PEtabModel(rn, observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_problem_rn.nllh(get_x(petab_problem_rn))
nll_sys = petab_problem_sys.nllh(get_x(petab_problem_sys))
@test nll_rn ≈ 0.75799668259765 atol=1e-3
@test nll_sys ≈ 0.75799668259765 atol=1e-3
