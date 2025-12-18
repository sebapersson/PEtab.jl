#=
    Test 0017 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS17 begin
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
@mtkbuild sys = SYS17()

measurements = DataFrame(simulation_id=["c0", "c0"],
                         pre_eq_id = ["preeq_c0", "preeq_c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[1.0, 10.0],
                         measurement=[0.7, 0.1])

@unpack k1, A, B = rn
simulation_conditions = [PEtabCondition(:preeq_c0, [k1, A, B], [0.3, 0.0, 2.0]),
                         PEtabCondition(:c0, [:k1, :A, :B], [0.8, 1.0, NaN])]

parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

model_rn = PEtabModel(rn, observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_problem_rn.nllh(get_x(petab_problem_rn))
nll_sys = petab_problem_sys.nllh(get_x(petab_problem_sys))
@test nll_rn ≈ 1.22063957624351 atol=1e-3
@test nll_sys ≈ 1.22063957624351 atol=1e-3
