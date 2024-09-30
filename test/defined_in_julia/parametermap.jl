#=
    Extra test to verify setting constant parameters work
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS begin
    @parameters begin
        a0
        b0
        k1
        k2 = 1.6
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
@mtkbuild sys = SYS()

parametermap = [:k2 => 1.6]

measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         noise_parameters=0.5)

simulation_conditions = Dict("c0" => Dict())

parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:log10)]

@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

model_rn = PEtabModel(rn, observables, measurements, parameters, parametermap=parametermap,
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn, verbose=false)
model_sys = PEtabModel(sys, observables, measurements, parameters;
                       simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys, verbose=false)

nll_rn = petab_problem_rn.nllh(petab_problem_rn.xnominal_transformed)
nll_sys = petab_problem_sys.nllh(petab_problem_sys.xnominal_transformed)
@test nll_rn ≈ 1.2738049275398435 atol=1e-3
@test nll_sys ≈ 1.2738049275398435 atol=1e-3
