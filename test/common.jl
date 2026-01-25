using Catalyst: @unpack
using SciMLBase

function _compute_nllh(x, model::PEtabModel, osolver::ODESolver; ss_solver = nothing)
    prob = PEtabODEProblem(
        model; odesolver = osolver, ss_solver = ss_solver
    )
    return prob.nllh(x)
end

function _compute_grad(
        x, model::PEtabModel, gradient_method, osolver::ODESolver;
        ss_solver = nothing, split = false, sensealg = :ForwardDiff
    )
    prob = PEtabODEProblem(
        model; odesolver = osolver, ss_solver = ss_solver,
        verbose = false, gradient_method = gradient_method,
        split_over_conditions = split, sensealg = sensealg
    )
    return prob.grad(x)
end

function _compute_hess(
        x, model::PEtabModel, hessian_method, osolver::ODESolver;
        ss_solver = nothing, split = false, sensealg = :ForwardDiff
    )
    prob = PEtabODEProblem(
        model; odesolver = osolver, ss_solver = ss_solver,
        verbose = false, hessian_method = hessian_method,
        split_over_conditions = split, sensealg = sensealg
    )
    _ = prob.nllh(x)
    return prob.hess(x)
end

# This end up being one way to check Gauss-Newton by checking that the gradient of the
# residuals (used by Gauss-Newton) is computed correctly
function test_grad_residuals(
        model::PEtabModel, osolver::ODESolver; ss_solver = nothing
    )::Nothing
    model_info = PEtab.ModelInfo(model, :ForwardDiff, nothing)
    probinfo = PEtab.PEtabODEProblemInfo(
        model, model_info, osolver, nothing, ss_solver,
        nothing, nothing, :GaussNewton, nothing, :ForwardDiff,
        nothing, false, false, SciMLBase.FullSpecialize,
        nothing, false, false
    )

    prior, _, hess_prior = PEtab._get_prior(model_info)
    residuals_sum = PEtab._get_nllh(probinfo, model_info, prior, true)
    _, jac_residuals = PEtab._get_hess(probinfo, model_info, hess_prior; ret_jacobian = true)

    xnames = model_info.xindices.ids[:estimate]
    xnames_ps = model_info.xindices.ids[:estimate_ps]
    x = PEtab._get_xnominal(model_info, xnames, xnames_ps, true)
    jac = zeros(length(x), length(model_info.petab_measurements.time))
    residual_grad = ForwardDiff.gradient(residuals_sum, x)
    jac = jac_residuals(x)
    @test all(≈(sum(jac, dims = 2), residual_grad, atol = 1.0e-6))
    return nothing
end
