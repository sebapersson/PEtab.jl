#=
    Test 0018 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS18 begin
    @parameters begin
        k1
        k2
    end
    @variables begin
        A(t)
        B(t)
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS18()

measurements = DataFrame(simulation_id=["c0", "c0", "c0", "c0"],
                         pre_eq_id = ["preeq_c0", "preeq_c0", "preeq_c0", "preeq_c0"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_b"],
                         time=[0.0, 1.0, 10.0, 0.0],
                         measurement=[0.1, 0.7, 0.1, 0.1])

simulation_conditions = [PEtabCondition(:preeq_c0, "k1" => 0.3, "A" => 0.0, "B" => 2.0),
                         PEtabCondition(:c0, :k1 => 0.8, :A => 1.0, :B => NaN)]

parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

@unpack A, B = rn
observables = [PEtabObservable("obs_a", A, 0.5),
               PEtabObservable("obs_b", B, 0.2)]

model_rn = PEtabModel(rn, observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_prob_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_prob_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_problem_rn.nllh(get_x(petab_problem_rn))
nll_sys = petab_problem_sys.nllh(get_x(petab_problem_sys))
@test nll_rn ≈ 6.3898204385477 atol=1e-3
@test nll_sys ≈ 6.3898204385477 atol=1e-3
