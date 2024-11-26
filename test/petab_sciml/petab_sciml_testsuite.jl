include(joinpath(@__DIR__, "helper.jl"))

# TODO: Refactor ids handling to be more clear
# TODO: Refactor transform to be more clear what is transformed (mech or nn)
# 9min
@testset "PEtab SciML extension" begin
    for i in 1:16
        test_case = i < 10 ? "00$i" : "0$i"
        path_yaml = joinpath(@__DIR__, "test_cases", test_case, "petab", "problem_ude.yaml")
        nnmodels = PEtab.load_nnmodels(path_yaml)
        osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10)
        model = PEtabModel(path_yaml; nnmodels = nnmodels, verbose = false)
        for config in PROB_CONFIGS
            # Edge case for unperforment configuration. So even though support could be
            # added in practice, it is not priority
            if (config.grad == :ForwardEquations && config.split == false &&
                config.sensealg == :ForwardDiff)
                continue
            end
            petab_prob = PEtabODEProblem(model; odesolver = osolver,
                                         gradient_method = config.grad,
                                         split_over_conditions = config.split,
                                         sensealg=config.sensealg)
            test_model(test_case, petab_prob)
        end
    end
end

petab_prob = PEtabODEProblem(model; odesolver = osolver, split_over_conditions = true, gradient_method = :ForwardDiff)
x = get_x(petab_prob)
petab_prob.nllh(x)
petab_prob.grad(x)
