include(joinpath(@__DIR__, "petab_sciml_testsuite", "helper.jl"))

dir_tests = joinpath(@__DIR__, "petab_sciml_testsuite", "test_cases")

@testset "PEtab SciML net import" begin
    for i in 1:53
        testcase = i < 10 ? "00$i" : "0$i"
        # ml_model must be loaded here to avoid world-problem
        dirtest = joinpath(dir_tests, "ml_model_import", "$testcase")
        yaml_test = YAML.load_file(joinpath(dirtest, "solutions.yaml"))
        ml_model, _ = PEtab.parse_to_lux(joinpath(dirtest, yaml_test["net_file"]))
        test_netimport(testcase, ml_model)
    end
end

ode_solver = ODESolver(Rodas5P(), abstol = 1e-10, reltol = 1e-10, maxiters=Int(1e6))
@testset "PEtab SciML hybrid models" begin
    for i in 1:34
        test_case = i < 10 ? "00$i" : "0$i"
        path_yaml = joinpath(
            dir_tests, "sciml_problem_import", test_case, "petab", "problem.yaml"
        )
        ml_models = MLModels(path_yaml)
        model = PEtabModel(path_yaml; ml_models = ml_models)
        for config in PROB_CONFIGS
            # Edge case for underperforming configuration. So even though support could be
            # added in theory, it is not priority
            if (config.grad == :ForwardEquations && config.split == false &&
                config.sensealg == :ForwardDiff)
                continue
            end
            # To long time, will not be used in practice
            if config.sensealg == ForwardSensitivity()
                continue
            end
            petab_prob = PEtabODEProblem(
                model; odesolver = ode_solver, gradient_method = config.grad,
                split_over_conditions = config.split, sensealg=config.sensealg
            )
            test_hybrid(test_case, petab_prob)
        end
    end
end

@testset "PEtab SciML initialization" begin
    for i in 1:3
        @info "Initialization test case $i"
        test_case = i < 10 ? "00$i" : "0$i"
        path_yaml = joinpath(
            dir_tests, "initialization", test_case, "petab", "problem.yaml"
        )
        ml_models = MLModels(path_yaml)
        model = PEtabModel(path_yaml; ml_models = ml_models, verbose = false)
        test_init(test_case, model)
    end
end

# PEtab SciML Julia interface
ode_solver = ODESolver(Rodas5P(), abstol = 1e-10, reltol = 1e-10, maxiters=Int(1e6))
@testset "SciML model in Julia" begin
    for i in 1:34
        test_case = i < 10 ? "00$(i).jl" : "0$(i).jl"
        include(joinpath(@__DIR__, "petab_sciml_testsuite", test_case))
    end
end
