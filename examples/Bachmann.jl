#=
    Example for how to set a PEtab ODE problem for the medium sized Bachmann model, and how to compute the gradient
    via forward-sensitivity equations and adjoint sensitivity analysis. Further, we show given a parameter vector
    how to perform Hessian based optmization using the Gauss-Newton hessian approximation for the optmizer;
        1) Fides.py (Trust-region Newton for box-constrained problems)

    This example shows how to best setup a PEtab problem for a medium sized model (20 ≤ states ≤ 50, 20 ≤ parameters ≤ 75).

    Besides this in the example folder we also have:
    Boehm.jl - here we show how to best handle small models (states ≤ 20, parameters ≤ 20). We further cover more details
        about the important PEtabModel and PEtabODEProblem functions. Recommended to checkout before looking at
        Bachmann.jl, Beer.jl and Brannmark.jl
    Beer.jl - here we show how to handle models when majority of parameter are specific to specific experimental conditions.
    Brannmark.jl - here we show how to handle models with preequilibration (model must be simulated to steady-state)
=#

using PEtab
using OrdinaryDiffEq
using Sundials # For CVODE_BDF
using SciMLSensitivity
using Zygote
using Printf

# Create the PEtabModel
path_yaml = joinpath(@__DIR__, "Bachmann", "Bachmann_MSB2011.yaml") # @__DIR__ = file directory
petab_model = PEtabModel(path_yaml, verbose = true)

#=
    --- Adjoint sensitivity analysis ---
    For some medium sized models and definitely for big models gradients are most efficiently computed via adjoint sensitivity
    analysis. When choosing the gradient_method=:Adjoint there are several other options that affect performance the most
    important are;

    1) ode_solver_gradient - Which ODE solver and which solver tolerances (abstol and reltol) to use when computing the
       gradient (in this case when solving the adjoint system). Below we use CVODE_BDF() which currently it the best performing
       stiff solver for the adjoint problem in Julia default abstol=reltol=1e-8.
       Note - this defaults to the ODE solver used when computing the cost if unspecified.
    2) sensealg - which adjoint algorithm to use. Currently we support InterpolatingAdjoint and QuadratureAdjoint from
       SciMLSensitivity (see their documentation for info https://github.com/SciML/SciMLSensitivity.jl). The user can
       provide any of the options that InterpolatingAdjoint and QuadratureAdjoint accepts, so if you want to use the
       ReverseDiffVJP an acceptble option is; InterpolatingAdjoint(autojacvec=ReversDiffVJP())

    Note1 - currently adjoint sensitivity analysis is not as stable in Julia as in AMICI (https://github.com/SciML/SciMLSensitivity.jl/issues/795),
       but our benchmarks show that SciMLSensitivity has the potential to be faster than other softwares.
    Note2 - the compilation times can be quite hefty for adjoint sensitivity analysis.
    Note3 - below we use QNDF for the cost which often is one of the best Julia solvers for larger models.
=#
petab_problem = PEtabODEProblem(petab_model,
                                ode_solver = ODESolver(QNDF(), abstol = 1e-8,
                                                       reltol = 1e-8),
                                ode_solver_gradient = ODESolver(CVODE_BDF(), abstol = 1e-8,
                                                                reltol = 1e-8),
                                gradient_method = :Adjoint,
                                sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP())) # EnzymeVJP is fastest when applicble
p = petab_problem.θ_nominalT # Parameter values in the PEtab file on log-scale
gradient = zeros(length(p)) # In-place gradients
cost = petab_problem.compute_cost(p)
petab_problem.compute_gradient!(gradient, p)
@printf("Cost for Bachmann = %.2f\n", cost)
@printf("First element in the gradient for Bachmann adjoint sensitivity analysis = %.2e\n",
        gradient[1])

#=
    --- Forward sensitivity equations and Gauss-Newton ---
    For medium sized models where computing the full Hessian via forward-mode automatic differentitation is too expensive
    the Guass-Newton hessian approximation often performs better than the (L)-BFGS approximation. To compute the Gauss-Newton
    approximation we need the model sensitivites which are obtained via forward sensitivity equations. These sensitivites can
    also be used to compute the gradient. As some optmizers such as Fides.py compute both the hessian and gradient at each iteration
    it is useful if we can save the sensitivity matrix, and thus if we compute the gradient via forward sensitivity equations.
    When choosing the gradient_method=:ForwardEquations and hessian_method=:GaussNewton there are several other options that affect
    performance the most important are;

    1) sensealg - which sensitivity algorithm to use when solving for the sensitives. We support both ForwardSensitivity and
       ForwardDiffSensitivity() with tuneable options as provided by SciMLSensitivity (see their documentation for info
       https://github.com/SciML/SciMLSensitivity.jl). The most efficient option though is :ForwardDiff where forward mode
       automatic differentitation is used to compute the sensitvites.
    2) reuse_sensitivities::Bool - whether or not to reuse the sensitives from the gradient compuations when computing the Gauss-Newton
       hessian-approximation. Whether this option is applicble or not depends on the optimizers, for example it works with
       Fides.py but not Optim.jl:s IPNewton(). When applicble it greatly reduces run-time.
       Note - this approach requires that sensealg=:ForwardDiff for the gradient.
=#
petab_problem = PEtabODEProblem(petab_model,
                                ode_solver = ODESolver(QNDF(), abstol = 1e-8,
                                                       reltol = 1e-8),
                                gradient_method = :ForwardEquations,
                                hessian_method = :GaussNewton,
                                sensealg = :ForwardDiff, # Fastest by far for computing the sensitivity matrix
                                reuse_sensitivities = true)
p = petab_problem.θ_nominalT # Parameter values in the PEtab file on log-scale
gradient = zeros(length(p)) # In-place gradients
hessian = zeros(length(p), length(p)) # In-place hessians
cost = petab_problem.compute_cost(p)
petab_problem.compute_gradient!(gradient, p)
petab_problem.compute_hessian!(hessian, p)
@printf("Cost for Bachmann = %.2f\n", cost)
@printf("First element in the gradient for Bachmann forward sensitivity equations = %.2e\n",
        gradient[1])
@printf("First element in the Gauss-Newton hessian for Bachmann = %.2f\n", hessian[1, 1])
