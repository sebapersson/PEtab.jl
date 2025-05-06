include(joinpath(@__DIR__, "helper.jl"))

@testset "PEtab SciML net import" begin
    for i in 1:51
        testcase = i < 10 ? "00$i" : "0$i"
        # nnmodel must be loaded here to avoid world-problem
        dirtest = joinpath(@__DIR__, "test_cases", "net_import", "$testcase")
        yaml_test = YAML.load_file(joinpath(dirtest, "solutions.yaml"))
        nnmodel, _ = PEtab.parse_to_lux(joinpath(dirtest, yaml_test["net_file"]))
        test_netimport(testcase, nnmodel)
    end
end

@testset "PEtab SciML hybrid models" begin
    for i in 1:25
        test_case = i < 10 ? "00$i" : "0$i"
        path_yaml = joinpath(@__DIR__, "test_cases", "hybrid", test_case, "petab", "problem.yaml")
        nnmodels = PEtab.load_nnmodels(path_yaml)
        osolver = ODESolver(Rodas5P(autodiff = false), abstol = 1e-10, reltol = 1e-10, maxiters=Int(1e6))
        model = PEtabModel(path_yaml; nnmodels = nnmodels, verbose = false)
        for config in PROB_CONFIGS
            # Edge case for underperforming configuration. So even though support could be
            # added in theory, it is not priority
            if (config.grad == :ForwardEquations && config.split == false &&
                config.sensealg == :ForwardDiff)
                continue
            end
            # To long time, will not be used in practice
            if config.sensealg == ForwardSensitivity() && test_case == "010"
                continue
            end
            petab_prob = PEtabODEProblem(model; odesolver = osolver,
                                         gradient_method = config.grad,
                                         split_over_conditions = config.split,
                                         sensealg=config.sensealg)
            test_hybrid(test_case, petab_prob)
        end
    end
end

@testset "PEtab SciML initialization" begin
    for i in 1:3
        @info "Initialization test case $i"
        test_case = i < 10 ? "00$i" : "0$i"
        path_yaml = joinpath(@__DIR__, "test_cases", "initialization", test_case, "petab",
                             "problem.yaml")
        nnmodels = PEtab.load_nnmodels(path_yaml)
        model = PEtabModel(path_yaml; nnmodels = nnmodels, verbose = false)
        test_init(test_case, model)
    end
end
