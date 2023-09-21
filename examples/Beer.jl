#= 
    In this example we setup the PEtab problem with the best options for a small ODE-model (≤20 states, ≤20 parameters) but with 
    many parameters to estimate (≈70) because several parameter are specific to experimental condition. For example, in 
    cond1 we have τ_cond1 and in cond2 we have τ_cond2, and both map to the ODE-system parameter τ. 
    
    Besides this in the example folder we also have:
    Boehm.jl - here we show how to best handle small models (states ≤ 20, parameters ≤ 20). We further cover more details 
        about the important PEtabModel and PEtabODEProblem functions. Recommended to checkout before looking at
        Bachmann.jl, Beer.jl and Brannmark.jl.
    Bachmann.jl - here we show how to set the best options for a medium sized model (20 ≤ states ≤ 50, 20 ≤ parameters ≤ 75), 
        and how to compute gradients via adjoint sensitivity analysis.
    Brannmark.jl - here we show how to handle models with preequilibration (model must be simulated to steady-state)
=#

using PEtab
using OrdinaryDiffEq
using Printf

path_yaml = joinpath(@__DIR__, "Beer", "Beer_MolBioSystems2014.yaml") # @__DIR__ = file directory
petab_model = PEtabModel(path_yaml, verbose=true)

#=
    Given a PEtab model we can create a PEtabODEProblem. For a small ODE-system like Beer the most efficient gradient 
    method if ForwardDiff, and we can also compute the hessian via :ForwardDiff. However, as there are several condition 
    specific parameters and by default we compute the gradient and hessian via a single call to ForwardDiff.jl we will 
    have to do as many forward-passes (solve the ODE model) as there are model-parameters even though majority of the 
    parameters are not present for most conditions. Too force several ForwardDiff calls we can use the option 
    split_over_conditions=true. Overall, the most important options to set are;
    
    1) ODE-solver options - Which ODE solver and which solver tolerances (abstol and reltol). Below we use the ODE solver
       Rodas5P() (works well for smaller models ≤ 15 states), and we use the default abstol=reltol=1e-8. 
    2) Gradient option - Which gradient option to use (full list in documentation). For small models like Beer forward 
       mode automatic differentitation (AD) is fastest, thus below we choose :ForwardDiff.
    3) Hessian option - Which gradient option to use (full list in documentation). For small models like Beer with 
       ≤20 parameters it is computationally feasible to compute the fill Hessian via forward-mode AD. Thus below 
       we choose :ForwardDiff. 
    4) split_over_conditions - Force a call to ForwardDiff.jl per experimental condition. Most efficent for models where 
       only a minority of the total model parameters are present per experimental condition, else should never be used. 

    Note - For :ForwardDiff the user can set the Chunk-size (see https://juliadiff.org/ForwardDiff.jl/stable/). This can improve performance, 
    and we plan to add automatic tuning of it.
=#
petab_problem = PEtabODEProblem(petab_model, 
                                ode_solver=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8), 
                                ode_solver_gradient=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8),
                                gradient_method=:ForwardDiff, 
                                hessian_method=:ForwardDiff, 
                                split_over_conditions=true, 
                                sparse_jacobian=false)

p = petab_problem.θ_nominalT # Parameter values in the PEtab file on log-scale
gradient = zeros(length(p))
hessian = zeros(length(p), length(p))
cost = petab_problem.compute_cost(p)
petab_problem.compute_gradient!(gradient, p)
petab_problem.compute_hessian!(hessian, p)
@printf("Cost for Beer = %.2f\n", cost)
@printf("First element in the gradient for Beer = %.2e\n", gradient[1])
@printf("First element in the hessian for Beer = %.2f\n", hessian[1, 1])
