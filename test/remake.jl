using CSV, DataFrames, LinearAlgebra, OrdinaryDiffEqRosenbrock, PEtab, Test

function test_petab_remake(model::PEtabModel, xchange, what_check; testtol = 1e-3)
    ode_solver = ODESolver(Rodas5P(), abstol=1e-12, reltol=1e-12)
    if :GradientForwardDiff in what_check
        prob1 = PEtabODEProblem(model; odesolver=ode_solver, chunksize=2, verbose=false)
        prob2 = remake(prob1, xchange)

        # Fix set xest for other functions
        nllh1 = prob1.nllh(prob1.xnominal_transformed)
        nllh2 = prob2.nllh(prob2.xnominal_transformed)
        @test  nllh1 ≈ nllh2

        imatch = findall(in(prob2.xnames), prob1.xnames)
        g1 = prob1.grad(prob1.xnominal_transformed)
        g2 = prob2.grad(prob2.xnominal_transformed)
        @test all(.≈(g1[imatch], g2, atol = testtol))
    end

    if :GradientForwardEquations in what_check
        prob1 = PEtabODEProblem(model; odesolver=ode_solver, chunksize=2, gradient_method=:ForwardEquations, sensealg=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        g1 = prob1.grad(prob1.xnominal_transformed)
        g2 = prob2.grad(prob2.xnominal_transformed)
        @test all(.≈(g1[imatch], g2, atol = testtol))
    end

    if :GaussNewton in what_check
        prob1 = PEtabODEProblem(model; odesolver=ode_solver, chunksize=2, gradient_method=:ForwardDiff, hessian_method=:GaussNewton, sensealg=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        h1GN = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        _h1GN = prob1.hess(prob1.xnominal_transformed)
        h2GN = prob2.hess(prob2.xnominal_transformed)
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                h1GN[i1, j1] = _h1GN[i2, j2]
            end
        end
        @test all(.≈(h1GN, h2GN, atol = testtol))
    end

    if :Hessian in what_check
        prob1 = PEtabODEProblem(model; odesolver=ode_solver, chunksize=2, hessian_method=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        h1 = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        _ = prob1.nllh(prob1.xnominal_transformed) # ensure to allocate certain objects
        _h1 = prob1.hess(prob1.xnominal_transformed)
        h2 = prob2.hess(prob2.xnominal_transformed)
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                h1[i1, j1] = _h1[i2, j2]
            end
        end
        @test all(.≈(h1, h2, atol = testtol))
    end

    if :FIM in what_check
        prob1 = PEtabODEProblem(model; odesolver=ode_solver, chunksize=2, hessian_method=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        FIM1 = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        prob1.nllh(prob1.xnominal_transformed) # ensure to allocate certain objects
        _FIM1 = prob1.FIM(prob1.xnominal_transformed)
        FIM2 = prob2.FIM(prob2.xnominal_transformed)
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                FIM1[i1, j1] = _FIM1[i2, j2]
            end
        end
        @test all(.≈(FIM1, FIM2, atol = testtol))
    end
end

function test_remake_condition_ids(path_yaml, condition_ids_test, split_conditions::Bool)
    ode_solver = ODESolver(Rodas5P(); abstol = 1e-12, reltol = 1e-12)
    ss_solver = SteadyStateSolver(:Simulate; abstol=1e-9, reltol=1e-12, maxiters = Int(1e5))

    model_original = PEtabModel(path_yaml)
    prob_original1 = PEtabODEProblem(model_original; gradient_method = :ForwardDiff,
        hessian_method = :ForwardDiff, odesolver = ode_solver, ss_solver = ss_solver,
        split_over_conditions = split_conditions)
    prob_original2 = PEtabODEProblem(model_original; gradient_method = :ForwardEquations,
        hessian_method = :GaussNewton, odesolver = ode_solver, ss_solver = ss_solver,
        split_over_conditions = split_conditions)

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
            conditions_df = filter(row -> row.conditionId in string.(conditions), conditions_df_original)
        else
            _cid = reduce(vcat, collect.(conditions))
            conditions_df = filter(row -> row.conditionId in string.(_cid), conditions_df_original)
        end
        CSV.write(path_conditions, conditions_df; delim = '\t')

        measurements_df_original = CSV.read(path_measurements_tmp, DataFrame)
        if prob_original1.model_info.simulation_info.has_pre_equilibration == false
            measurements_df = filter(r -> r.simulationConditionId in string.(conditions), measurements_df_original)
        else
            pre_eq_ids = string.(getfield.(conditions, :pre_eq))
            simulation_ids = string.(getfield.(conditions, :simulation))
            experiment_ids = pre_eq_ids .* simulation_ids
            experiment_ids_df = measurements_df_original.preequilibrationConditionId .*
                measurements_df_original.simulationConditionId .|>
                string
            row_idx = findall(x -> x in experiment_ids, experiment_ids_df)
            measurements_df = measurements_df_original[row_idx, :]
        end
        CSV.write(path_measurements, measurements_df; delim = '\t')

        model_ref = PEtabModel(path_yaml)
        prob_ref1 = PEtabODEProblem(model_ref; gradient_method = :ForwardDiff,
            hessian_method = :ForwardDiff, odesolver = ode_solver, ss_solver = ss_solver,
            split_over_conditions = split_conditions)
        prob_ref2 = PEtabODEProblem(model_ref; gradient_method = :ForwardEquations,
            hessian_method = :GaussNewton, odesolver = ode_solver, ss_solver = ss_solver,
            split_over_conditions = split_conditions)
        x_ref = get_x(prob_ref1)

        prob_remade1 = remake(prob_original1; condition_ids=conditions)
        prob_remade2 = remake(prob_original2; condition_ids=conditions)
        x_remade = get_x(prob_remade1)

        @test prob_ref1.nllh(x_ref) ≈ prob_remade1.nllh(x_remade) atol = 1e-8
        @test prob_ref2.nllh(x_ref) ≈ prob_remade2.nllh(x_remade) atol = 1e-8

        # Order of parameters may differ between the problems
        ix = sortperm(prob_remade1.xnames)[invperm(sortperm(prob_ref1.xnames))]
        x_ref = collect(x_ref)
        x_remade = collect(x_remade)
        @test all(.≈(prob_ref1.grad(x_ref), prob_remade1.grad(x_remade)[ix]; atol = 1e-8))
        @test all(.≈(prob_ref2.grad(x_ref), prob_remade2.grad(x_remade)[ix]; atol = 1e-8))

        h_ref1 = prob_ref1.hess(x_ref)
        h_ref2 = prob_ref2.hess(x_ref)
        h_remade1 = prob_remade1.hess(x_remade)
        h_remade2 = prob_remade2.hess(x_remade)
        for (i1, i2) in pairs(ix)
            for (j1, j2) in pairs(ix)
                @test h_ref1[i1, j1] ≈ h_remade1[i2, j2] atol=1e-6
                @test h_ref2[i1, j1] ≈ h_remade2[i2, j2] atol=1e-6
            end
        end
    end

    mv(path_conditions_tmp, path_conditions; force = true)
    mv(path_measurements_tmp, path_measurements; force = true)
    return nothing
end

@info "Testing remake for Boehm model"
path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml, build_julia_files=true, verbose=false, write_to_file = false)
xchange1 = Dict(:k_imp_hetero => 0.0163679184468,
                :k_exp_homo => 0.006170228086381,
                :k_phos => 15766.5070195731)
xchange2 = Dict(:k_imp_hetero => "estimate",
                :k_exp_homo => 0.006170228086381,
                :k_phos => "estimate")
xchange3 = Dict(:sd_pSTAT5A_rel => 3.85261197844677)
xchange4 = Dict(:sd_pSTAT5A_rel => 3.85261197844677,
                :k_exp_homo => 0.006170228086381)
@testset "PEtab remake : Boehm" begin
    methods_test = [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian,
                    :FIM]
    test_petab_remake(model, xchange1, methods_test)
    test_petab_remake(model, xchange2, methods_test)
    test_petab_remake(model, xchange3, methods_test)
    test_petab_remake(model, xchange4, methods_test)
end

# Test for Brannmark (has steady state simulations)
@info "Testing remake for Brannmark model"
path_yaml = joinpath(@__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
model = PEtabModel(path_yaml,  build_julia_files=true, verbose=false, write_to_file = false)
xchange1 = Dict(:k1a => 0.177219477727669,
                :k22 => 666.8355739795,
                :km2 => 1.15970741690448)
xchange2 = Dict(:k1a => "estimate",
                :k22 => 666.8355739795,
                :km2 => "estimate")
xchange3 = Dict(:sigmaY2Step => 5.15364156671777)
xchange4 = Dict(:sigmaY2Step => 5.15364156671777,
                :k22 => 666.8355739795)
@testset "PEtab remake : Brannmark" begin
    methods_test = [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton]
    test_petab_remake(model, xchange1, methods_test; testtol = 1e-2)
    test_petab_remake(model, xchange2, methods_test; testtol = 1e-2)
    test_petab_remake(model, xchange3, methods_test; testtol = 1e-2)
    test_petab_remake(model, xchange4, methods_test; testtol = 1e-2)
end

path_yaml = joinpath(@__DIR__, "published_models", "Bruno_JExpBot2016", "Bruno_JExpBot2016.yaml")
@testset begin "PEtab remake: Bruno condition ids"
    condition_ids_test = [
        [:model1_data1, :model1_data2, :model1_data4],
    ]
    test_remake_condition_ids(path_yaml, condition_ids_test, false)
    test_remake_condition_ids(path_yaml, condition_ids_test, true)

    prob = PEtabModel(path_yaml) |>
        PEtabODEProblem
    prob_remade = remake(prob)
    @test prob.nllh(get_x(prob)) == prob_remade.nllh(get_x(prob))

    condition_ids = [(pre_eq = :Dose_0, simulation = :Dose_0)]
    @test_throws PEtab.PEtabFormatError remake(prob; condition_ids=condition_ids)
    @test_throws PEtab.PEtabFormatError remake(prob; condition_ids=[:hej])
end

path_yaml = joinpath(@__DIR__, "published_models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
@testset "PEtab remake: Brannmark condition ids" begin
    condition_ids_test = [
        [(pre_eq = :Dose_0, simulation = :Dose_0),
         (pre_eq = :Dose_0, simulation = :Dose_1),
         (pre_eq = :Dose_0, simulation = :Dose_100)]
    ]
    test_remake_condition_ids(path_yaml, condition_ids_test, false)

    prob = PEtabModel(path_yaml) |>
        PEtabODEProblem
    prob_remade = remake(prob; condition_ids=NamedTuple[])
    @test prob.nllh(get_x(prob)) == prob_remade.nllh(get_x(prob))
    @test_throws PEtab.PEtabFormatError remake(prob; condition_ids=[:hej])
    @test_throws PEtab.PEtabFormatError remake(prob; condition_ids=[(pre_eq = :Dose_1, simulation = :Dose_0)])
end
