using PEtab, Test, OrdinaryDiffEq, LinearAlgebra

function test_petab_remake(model::PEtabModel, xchange, what_check)
    if :GradientForwardDiff in what_check
        prob1 = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P()), chunksize=2, verbose=false)
        prob2 = remake(prob1, xchange)

        # Fix set xest for other functions
        nllh1 = prob1.nllh(prob1.xnominal_transformed)
        nllh2 = prob2.nllh(prob2.xnominal_transformed)
        @test  nllh1 ≈ nllh2

        imatch = findall(in(prob2.xnames), prob1.xnames)
        g1 = zeros(length(prob1.xnominal_transformed))
        g2 = zeros(length(prob2.xnominal_transformed))
        prob1.grad!(g1, prob1.xnominal_transformed)
        prob2.grad!(g2, prob2.xnominal_transformed)
        @test all(.≈(g1[imatch, g2], atol = 1e-3))
    end

    if :GradientForwardEquations in what_check
        prob1 = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P()), chunksize=2, gradient_method=:ForwardEquations, sensealg=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        g1 = zeros(length(prob1.xnominal_transformed))
        g2 = zeros(length(prob2.xnominal_transformed))
        prob1.grad!(g1, prob1.xnominal_transformed)
        prob2.grad!(g2, prob2.xnominal_transformed)
        @test all(.≈(g1[imatch, g2], atol = 1e-3))
    end

    if :GaussNewton in what_check
        prob1 = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P()), chunksize=2, gradient_method=:ForwardDiff, hessian_method=:GaussNewton, sensealg=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        _h1GN = zeros(length(prob1.xnominal_transformed), length(prob1.xnominal_transformed))
        h1GN = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        h2GN = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        prob1.hess!(_h1GN, prob1.xnominal_transformed)
        prob2.hess!(h2GN, prob2.xnominal_transformed)
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                h1GN[i1, j1] = _h1GN[i2, j2]
            end
        end
        @test all(.≈(h1GN, h2GN, atol = 1e-3))
    end

    if :Hessian in what_check
        prob1 = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P()), chunksize=2, hessian_method=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        _h1 = zeros(length(prob1.xnominal_transformed), length(prob1.xnominal_transformed))
        h1 = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        h2 = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        prob1.nllh(prob1.xnominal_transformed) # ensure to allocate certain objects
        prob1.hess!(_h1, prob1.xnominal_transformed)
        prob2.hess!(h2, prob2.xnominal_transformed)
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                h1[i1, j1] = _h1[i2, j2]
            end
        end
        @test all(.≈(h1, h2, atol = 1e-3))
    end

    if :FIM in what_check
        prob1 = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P()), chunksize=2, hessian_method=:ForwardDiff, verbose=false)
        prob2 = remake(prob1, xchange)

        imatch = findall(in(prob2.xnames), prob1.xnames)
        _FIM1 = zeros(length(prob1.xnominal_transformed), length(prob1.xnominal_transformed))
        FIM1 = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        FIM2 = zeros(length(prob2.xnominal_transformed), length(prob2.xnominal_transformed))
        prob1.nllh(prob1.xnominal_transformed) # ensure to allocate certain objects
        prob1.FIM!(_FIM1, prob1.xnominal_transformed)
        prob2.FIM!(FIM2, prob2.xnominal_transformed)
        for (i1, i2) in pairs(imatch)
            for (j1, j2) in pairs(imatch)
                FIM1[i1, j1] = _FIM1[i2, j2]
            end
        end
        @test all(.≈(FIM1, FIM2, atol = 1e-3))
    end
end


# Test for Boehm (no steady-state simulations)
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
    test_petab_remake(model, xchange1, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
    test_petab_remake(model, xchange2, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
    test_petab_remake(model, xchange3, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
    test_petab_remake(model, xchange4, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton, :Hessian, :FIM])
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
    test_petab_remake(model, xchange1, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    test_petab_remake(model, xchange2, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    test_petab_remake(model, xchange3, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
    test_petab_remake(model, xchange4, [:GradientForwardDiff, :GradientForwardEquations, :GaussNewton])
end
