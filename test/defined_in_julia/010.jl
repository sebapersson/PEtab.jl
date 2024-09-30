#=
    Test 0010 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS begin
    @parameters begin
        k1
        k2
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
@mtkbuild sys = SYS()

speciemap = [:A => 1.0]

measurements = DataFrame(pre_eq_id=["preeq_c0", "preeq_c0"],
                         simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[1.0, 10.0],
                         measurement=[0.7, 0.1])

simulation_conditions = Dict("c0" => Dict(:k1 => 0.8, :B => 1.0),
                             "preeq_c0" => Dict(:k1 => 0.3, :B => 0.0))

parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, :lin, 0.5))

model_rn = PEtabModel(sys, observables, measurements, parameters; speciemap = speciemap,
                      simulation_conditions = simulation_conditions, )
petab_problem_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_problem_rn.nllh(petab_problem_rn.xnominal_transformed)
nll_sys = petab_problem_sys.nllh(petab_problem_sys.xnominal_transformed)
@test nll_rn ≈ 1.20628941926143 atol=1e-3
@test nll_sys ≈ 1.20628941926143 atol=1e-3
