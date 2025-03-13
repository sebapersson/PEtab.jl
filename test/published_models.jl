#=
    Check the accruacy of the PeTab importer by checking the log-likelihood value against known values for several
    published models. Also check gradients for selected models using FiniteDifferences package
=#

using PEtab, OrdinaryDiffEqRosenbrock, SciMLSensitivity, LinearAlgebra, FiniteDifferences,
    Sundials, Test

include(joinpath(@__DIR__, "common.jl"))

NLLH_MODELS = Dict(
    :Bachmann_MSB2011 => (nllh = -418.40573341425295, prior = 0.0),
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

GRAD_MODELS = Dict(
    :Bachmann_MSB2011 => (test=[:forward_AD, :forward_eqs, :forward_eqs_sciml, :adjoint], tol=1e-2, odetol=1e-9, split = false),
    :Beer_MolBioSystems2014 => (test=[:forward_AD, :forward_eqs], tol=1e-1, odetol = 1e-8, split = true),
    :Brannmark_JBC2010 => (test=[:forward_AD, :forward_eqs], tol=2e-3, odetol = 1e-10, split = false),
    :Bruno_JExpBot2016 => (test=[:forward_AD, :forward_eqs], tol=1e-1, odetol = 1e-8, split = false),
    :Schwen_PONE2014 => (test=[:forward_AD, :forward_eqs], tol=1e-3, odetol = 1e-8, split = false))

function test_nllh(modelid::Symbol)::Nothing
    @info "nllh model $modelid"
    path = joinpath(@__DIR__, "published_models", "$modelid", "$(modelid).yaml")
    model = PEtabModel(path)
    osolver = ODESolver(Rodas5P(), abstol = 1e-10, reltol = 1e-10, maxiters = Int(1e5))
    ssolver = SteadyStateSolver(:Simulate; abstol=5e-10, reltol=1e-10, maxiters = Int(1e5))
    prob = PEtabODEProblem(model; odesolver = osolver, ss_solver = ssolver,
                           sparse_jacobian = false)
    nllh = prob.nllh(get_x(prob))
    nllh_ref = NLLH_MODELS[modelid].nllh
    prior_ref = NLLH_MODELS[modelid].prior
    @test nllh ≈ nllh_ref + prior_ref atol = 1e-2
    return nothing
end

function test_grad(modelid::Symbol)::Nothing
    @info "Gradient model $modelid"
    path = joinpath(@__DIR__, "published_models", "$modelid", "$(modelid).yaml")
    model = PEtabModel(path)

    odetol = GRAD_MODELS[modelid].odetol
    grads_test = GRAD_MODELS[modelid].test
    testtol = GRAD_MODELS[modelid].tol
    split = GRAD_MODELS[modelid].split

    ss_solver = SteadyStateSolver(:Simulate; abstol=1e-9, reltol=1e-12, maxiters = Int(1e5))
    osolver1 = ODESolver(Rodas5P(), abstol = odetol, reltol = odetol, maxiters = Int(1e5))
    osolver2 = ODESolver(CVODE_BDF(), abstol = odetol, reltol = odetol)
    prob_ref = PEtabODEProblem(model; odesolver = osolver1, ss_solver = ss_solver,
                               verbose = false)
    x = get_x(prob_ref)
    prob_ref.nllh(x)
    gref = FiniteDifferences.grad(central_fdm(5, 1), prob_ref.nllh, x)[1]

    for grad_test in grads_test
        @info "Method $(grad_test)"
        if grad_test == :forward_eqs_sciml
            g = _compute_grad(x, model, :ForwardEquations, osolver2;
                              ss_solver = ss_solver, sensealg = ForwardSensitivity())
        elseif grad_test == :forward_eqs
            g = _compute_grad(x, model, :ForwardEquations, osolver1;
                              ss_solver = ss_solver, split = split)
        elseif grad_test == :adjoint
            g = _compute_grad(x, model, :Adjoint, osolver1;  ss_solver = ss_solver,
                              sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        else
            g = _compute_grad(x, model, :ForwardDiff, osolver1; ss_solver = ss_solver,
                              split = split)
        end
        @test all(.≈(gref, g; atol = testtol))
    end
    return nothing
end

for model in keys(NLLH_MODELS)
    test_nllh(model)
end

for model in keys(GRAD_MODELS)
    test_grad(model)
end
