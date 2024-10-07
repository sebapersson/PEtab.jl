#=
    Check the accruacy of the PeTab importer by checking the log-likelihood value against known values for several
    published models. Also check gradients for selected models using FiniteDifferences package
=#

using PEtab, OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, LinearAlgebra, FiniteDifferences,
      Sundials, Test

include(joinpath(@__DIR__, "common.jl"))

NLLH_MODELS = Dict(:Bachmann_MSB2011 => (nllh = -418.40573341425295, prior = 0.0),
                   :Beer_MolBioSystems2014 => (nllh = -58622.9145631413, prior = 0.0),
                   :Bruno_JExpBot2016 => (nllh = -46.688181449443945, prior = 0.0),
                   :Brannmark_JBC2010 => (nllh = 141.889113770537, prior = 0.0),
                   :Crauste_CellSystems2017 => (nllh = 190.96521897435176, prior = 0.0),
                   :Fujita_SciSignal2010 => (nllh = -53.08377736998929, prior = 0.0),
                   :Isensee_JCB2018 => (nllh = 3949.375966548649, prior = 4.45299970460275),
                   :Sneyd_PNAS2002 => (nllh = -319.79177818768756, prior = 0.0),
                   :Zheng_PNAS2012 => (nllh = -278.33353271001477, prior = 0.0),
                   :Schwen_PONE2014 => (nllh = 943.9992988598723, prior = 12.519137073132825),
                   :Smith_BMCSystBiol2013 => (nllh = 343830.6310470444, prior = 0.0))

GRAD_MODELS = Dict(:Bachmann_MSB2011 => (test=[:forward_AD, :forward_eqs, :forward_eqs_sciml, :adjoint], tol=1e-2, odetol=1e-9, split = false),
                   :Beer_MolBioSystems2014 => (test=[:forward_AD, :forward_eqs], tol=1e-1, odetol = 1e-8, split = true),
                   :Bruno_JExpBot2016 => (test=[:forward_AD, :forward_eqs], tol=1e-1, odetol = 1e-8, split = false),
                   :Brannmark_JBC2010 => (test=[:forward_AD, :forward_eqs], tol=2e-3, odetol = 1e-10, split = false),
                   :Schwen_PONE2014 => (test=[:forward_AD, :forward_eqs], tol=1e-3, odetol = 1e-8, split = false))

function test_nllh(model::Symbol)::Nothing
    @info "nllh model $model"
    path = joinpath(@__DIR__, "published_models", "$model", "$(model).yaml")
    petab_model = PEtabModel(path; verbose = false, build_julia_files = true,
                             write_to_file = false)
    osolver = ODESolver(Rodas4P(), abstol = 1e-12, reltol = 1e-12, maxiters = Int(1e5))
    prob = PEtabODEProblem(petab_model; odesolver = osolver, verbose = false)
    nllh = prob.nllh(prob.xnominal_transformed)
    nllh_ref = NLLH_MODELS[model].nllh
    prior_ref = NLLH_MODELS[model].prior
    @test nllh ≈ nllh_ref + prior_ref atol = 1e-2
    return nothing
end

function test_grad(model::Symbol)::Nothing
    @info "Gradient model $model"
    path = joinpath(@__DIR__, "published_models", "$model", "$(model).yaml")
    petab_model = PEtabModel(path; verbose = false, build_julia_files = true,
                             write_to_file = false)

    odetol = GRAD_MODELS[model].odetol
    grads_test = GRAD_MODELS[model].test
    testtol = GRAD_MODELS[model].tol
    split = GRAD_MODELS[model].split

    ss_solver = SteadyStateSolver(:Simulate; abstol=1e-12, reltol=1e-10)
    osolver1 = ODESolver(Rodas5P(), abstol = odetol, reltol = odetol)
    osolver2 = ODESolver(CVODE_BDF(), abstol = odetol, reltol = odetol)
    prob_ref = PEtabODEProblem(petab_model; odesolver = osolver1, ss_solver = ss_solver,
                               verbose = false)
    x = prob_ref.xnominal_transformed
    gref = FiniteDifferences.grad(central_fdm(5, 1), prob_ref.nllh, x)[1]

    for grad_test in grads_test
        @info "Method $(grad_test)"
        if grad_test == :forward_eqs_sciml
            g = _compute_grad(x, petab_model, :ForwardEquations, osolver2;
                              ss_solver = ss_solver, sensealg = ForwardSensitivity())
        elseif grad_test == :forward_eqs
            g = _compute_grad(x, petab_model, :ForwardEquations, osolver1;
                              ss_solver = ss_solver, split = split)
        elseif grad_test == :adjoint
            g = _compute_grad(x, petab_model, :Adjoint, osolver1;  ss_solver = ss_solver,
                              sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        else
            g = _compute_grad(x, petab_model, :ForwardDiff, osolver1; ss_solver = ss_solver,
                              split = split)
        end
        @test norm(gref - g) ≤ testtol
    end
    return nothing
end

for model in keys(NLLH_MODELS)
    test_nllh(model)
end

for model in keys(GRAD_MODELS)
    test_grad(model)
end
