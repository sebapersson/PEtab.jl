using CSV, DataFrames, LinearAlgebra, OrdinaryDiffEqRosenbrock, PEtab, Test

function test_remake_parameters(model::PEtabModel, xchange, what_check; testtol = 1.0e-3)
    ode_solver = ODESolver(Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12)
    if :GradientForwardDiff in what_check
        prob1 = PEtabODEProblem(model; odesolver = ode_solver)
        prob2 = remake(prob1; parameters = xchange)

        # Fix set xest for other functions
        nllh1 = prob1.nllh(get_x(prob1))
        nllh2 = prob2.nllh(get_x(prob2))
        @test  nllh1 ≈ nllh2

        imatch = findall(in(prob2.xnames), prob1.xnames)
        g1 = prob1.grad(get_x(prob1))
        g2 = prob2.grad(get_x(prob2))
        @test all(.≈(g1[imatch], g2, atol = testtol))
    end

    if :GradientForwardEquations in what_check
        prob1 = PEtabODEProblem(
            model; odesolver = ode_solver,
            gradient_method = :ForwardEquations, sensealg = :ForwardDiff
        )
        prob2 = remake(prob1; parameters = xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        g1 = prob1.grad(get_x(prob1))
        g2 = prob2.grad(get_x(prob2))
        @test all(.≈(g1[imatch], g2, atol = testtol))
    end

    if :GaussNewton in what_check
        prob1 = PEtabODEProblem(
            model; odesolver = ode_solver, gradient_method = :ForwardDiff,
            hessian_method = :GaussNewton, sensealg = :ForwardDiff
        )
        prob2 = remake(prob1; parameters = xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        h1GN = zeros(length(get_x(prob2)), length(get_x(prob2)))
        _h1GN = prob1.hess(get_x(prob1))
        h2GN = prob2.hess(get_x(prob2))
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                h1GN[i1, j1] = _h1GN[i2, j2]
            end
        end
        @test all(.≈(h1GN, h2GN, atol = testtol))
    end

    if :Hessian in what_check
        prob1 = PEtabODEProblem(
            model; odesolver = ode_solver, hessian_method = :ForwardDiff
        )
        prob2 = remake(prob1; parameters = xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        h1 = zeros(length(get_x(prob2)), length(get_x(prob2)))
        _ = prob1.nllh(get_x(prob1)) # ensure to allocate certain objects
        _h1 = prob1.hess(get_x(prob1))
        h2 = prob2.hess(get_x(prob2))
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                h1[i1, j1] = _h1[i2, j2]
            end
        end
        @test all(.≈(h1, h2, atol = testtol))
    end

    return if :FIM in what_check
        prob1 = PEtabODEProblem(
            model; odesolver = ode_solver, hessian_method = :ForwardDiff,
            verbose = false
        )
        prob2 = remake(prob1; parameters = xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        FIM1 = zeros(length(get_x(prob2)), length(get_x(prob2)))
        prob1.nllh(get_x(prob1)) # ensure to allocate certain objects
        _FIM1 = prob1.FIM(get_x(prob1))
        FIM2 = prob2.FIM(get_x(prob2))
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                FIM1[i1, j1] = _FIM1[i2, j2]
            end
        end
        @test all(.≈(FIM1, FIM2, atol = testtol))
    end
end

function test_remake_condition_ids(path_yaml, condition_ids_test, split_conditions::Bool)
    ode_solver = ODESolver(Rodas5P(); abstol = 1.0e-12, reltol = 1.0e-12)
    ss_solver = SteadyStateSolver(
        :Simulate; abstol = 1.0e-9, reltol = 1.0e-12, maxiters = Int(1.0e5)
    )

    model_original = PEtabModel(path_yaml)
    prob_original1 = PEtabODEProblem(
        model_original; gradient_method = :ForwardDiff,
        hessian_method = :ForwardDiff, odesolver = ode_solver, ss_solver = ss_solver,
        split_over_conditions = split_conditions
    )
    prob_original2 = PEtabODEProblem(
        model_original; gradient_method = :ForwardEquations,
        hessian_method = :GaussNewton, odesolver = ode_solver, ss_solver = ss_solver,
        split_over_conditions = split_conditions
    )

    dir_problem = prob_original1.model_info.model.paths[:dirmodel]
    path_conditions = prob_original1.model_info.model.paths[:conditions]
    path_conditions_tmp = joinpath(dir_problem, "conditions_tmp.tsv")
    cp(path_conditions, path_conditions_tmp; force = true)

    path_measurements = prob_original1.model_info.model.paths[:measurements]
    path_measurements_tmp = joinpath(dir_problem, "measurements.tsv")
    cp(path_measurements, path_measurements_tmp; force = true)

    for conditions in condition_ids_test
        conditions_df_original = CSV.read(path_conditions_tmp, DataFrame)
        if prob_original1.model_info.simulation_info.has_pre_equilibration == false
            conditions_df = filter(
                row -> row.conditionId in string.(conditions), conditions_df_original
            )
        else
            _cid = reduce(vcat, collect.(conditions))
            conditions_df = filter(
                row -> row.conditionId in string.(_cid), conditions_df_original
            )
        end
        CSV.write(path_conditions, conditions_df; delim = '\t')

        measurements_df_original = CSV.read(path_measurements_tmp, DataFrame)
        if prob_original1.model_info.simulation_info.has_pre_equilibration == false
            measurements_df = filter(
                r -> r.simulationConditionId in string.(conditions), measurements_df_original
            )
        else
            pre_eq_ids = string.(first.(conditions))
            simulation_ids = string.(last.(conditions))
            experiment_ids = pre_eq_ids .* simulation_ids
            experiment_ids_df = measurements_df_original.preequilibrationConditionId .*
                measurements_df_original.simulationConditionId .|>
                string
            row_idx = findall(x -> x in experiment_ids, experiment_ids_df)
            measurements_df = measurements_df_original[row_idx, :]
        end
        CSV.write(path_measurements, measurements_df; delim = '\t')

        model_ref = PEtabModel(path_yaml)
        prob_ref1 = PEtabODEProblem(
            model_ref; gradient_method = :ForwardDiff,
            hessian_method = :ForwardDiff, odesolver = ode_solver, ss_solver = ss_solver,
            split_over_conditions = split_conditions
        )
        prob_ref2 = PEtabODEProblem(
            model_ref; gradient_method = :ForwardEquations,
            hessian_method = :GaussNewton, odesolver = ode_solver, ss_solver = ss_solver,
            split_over_conditions = split_conditions
        )
        x_ref = get_x(prob_ref1)

        prob_remade1 = remake(prob_original1; conditions = conditions)
        prob_remade2 = remake(prob_original2; conditions = conditions)
        x_remade = get_x(prob_remade1)

        @test prob_ref1.nllh(x_ref) ≈ prob_remade1.nllh(x_remade) atol = 1.0e-8
        @test prob_ref2.nllh(x_ref) ≈ prob_remade2.nllh(x_remade) atol = 1.0e-8

        # Order of parameters may differ between the problems
        ix = sortperm(prob_remade1.xnames)[invperm(sortperm(prob_ref1.xnames))]
        x_ref = collect(x_ref)
        x_remade = collect(x_remade)
        @test all(.≈(prob_ref1.grad(x_ref), prob_remade1.grad(x_remade)[ix]; atol = 1.0e-8))
        @test all(.≈(prob_ref2.grad(x_ref), prob_remade2.grad(x_remade)[ix]; atol = 1.0e-8))

        h_ref1 = prob_ref1.hess(x_ref)
        h_ref2 = prob_ref2.hess(x_ref)
        h_remade1 = prob_remade1.hess(x_remade)
        h_remade2 = prob_remade2.hess(x_remade)
        for (i1, i2) in pairs(ix)
            for (j1, j2) in pairs(ix)
                @test h_ref1[i1, j1] ≈ h_remade1[i2, j2] atol = 1.0e-6
                @test h_ref2[i1, j1] ≈ h_remade2[i2, j2] atol = 1.0e-6
            end
        end
    end

    mv(path_conditions_tmp, path_conditions; force = true)
    mv(path_measurements_tmp, path_measurements; force = true)
    return nothing
end

path_yaml = joinpath(
    @__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"
)
model = PEtabModel(path_yaml)
xchange1 = [
    :k_imp_hetero => 0.0163679184468, :k_exp_homo => 0.006170228086381,
    :k_phos => 15766.5070195731,
]
xchange2 = [:k_exp_homo => 0.006170228086381]

@info "Remake parameters"
@testset "PEtab remake : Boehm parameters" begin
    methods_test = [
        :GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian,
        :FIM,
    ]
    test_remake_parameters(model, xchange1, methods_test)
    test_remake_parameters(model, xchange2, methods_test)
end

@info "Remake conditions Bruno"
path_yaml = joinpath(
    @__DIR__, "published_models", "Bruno_JExpBot2016", "Bruno_JExpBot2016.yaml"
)
@testset begin
    "PEtab remake: Bruno condition ids"
    condition_ids_test = [
        [:model1_data1, :model1_data2, :model1_data4],
    ]
    test_remake_condition_ids(path_yaml, condition_ids_test, false)
    test_remake_condition_ids(path_yaml, condition_ids_test, true)

    prob = PEtabModel(path_yaml) |>
        PEtabODEProblem
    prob_remade = remake(prob)
    @test prob.nllh(get_x(prob)) == prob_remade.nllh(get_x(prob))

    @test_throws PEtab.PEtabFormatError remake(prob; conditions = [:Dose_0 => :Dose_0])
    @test_throws PEtab.PEtabInputError remake(prob; conditions = [:hej])
end

@info "Remake conditions Brannmark"
path_yaml = joinpath(
    @__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml"
)
@testset "PEtab remake: Brannmark condition ids" begin
    condition_ids_test = [[:Dose_0 => :Dose_0, :Dose_0 => :Dose_1, :Dose_0 => :Dose_100]]

    test_remake_condition_ids(path_yaml, condition_ids_test, false)

    prob = PEtabModel(path_yaml) |>
        PEtabODEProblem
    @test_throws PEtab.PEtabFormatError remake(prob; conditions = [:hej])
    @test_throws PEtab.PEtabInputError remake(prob; conditions = [:Dose_1 => :Dose_0])
end

# Remake for v2 problems. As they rely on the funcionality as condition ids, only
# testing correct ids are removed
path_yaml = joinpath(@__DIR__, "petab_v2_testsuite", "0002", "_0002.yaml")
model = PEtabModel(path_yaml)
prob = PEtabODEProblem(model)
prob_remade = remake(prob; experiments = [:e1])
@test prob_remade.model_info.simulation_info.conditionids[:experiment] == [:e1_c0]

# Test errors
@test_throws ArgumentError remake(prob; conditions = [:e1])
prob = joinpath(
    @__DIR__, "published_models", "Bruno_JExpBot2016", "Bruno_JExpBot2016.yaml"
) |>
    PEtabModel |>
    PEtabODEProblem
@test_throws ArgumentError remake(prob; experiments = [:model1_data1])
@test_throws PEtab.PEtabInputError remake(prob, parameters = [:k3 => 3.0])
