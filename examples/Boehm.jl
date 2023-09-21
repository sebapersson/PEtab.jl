#= 
    In this example we setup the PEtab problem with the best options for a small model (≤20 states and ≤ 20 parameters). 
    
    Further examples include;
    Bachmann.jl - here we show how to set the best options for a medium sized model (20 ≤ states ≤ 50, 20 ≤ parameters ≤ 75), 
        and how to compute gradients via adjoint sensitivity analysis.
    Beer.jl - here we show how to handle models when majority of parameter are specific to specific experimental conditions.
=#

using PEtab
using OrdinaryDiffEq
using Printf

#= 
    Reading a PEtab model is easy. Given the path to the yaml-file PEtab.jl will read all PEtab files into a PEtabModel. 

    Several things happens under the hood when reading the PEtab model;
    1) The SBML file is translated into ModelingToolkit.jl format (e.g allow symbolic compuations of the 
       ODE-model Jacobian).
    2) The observables PEtab-table is transalted into Julia functions for computing the observable (h), 
       noise parameter (σ) and initial values (u0). 
    3) To be able to compute gradients via adjoint sensitivity analysis and/or forward sensitivity equations 
       of h and σ are computed symbolically with respect to the ODE-models states (u) and parameters. 
    
    All these files are created automatically, and and you can find the resulting files in dirIfYamlFile/Julia_files/. 
    To save time the fucntion PEtabModel has the default forceBuildJlFiles=false meaning that the Julia files 
    are not rebuilt in case they already exist.
=#    
path_yaml = joinpath(@__DIR__, "Boehm", "Boehm_JProteomeRes2014.yaml") # @__DIR__ = file directory
petab_model = PEtabModel(path_yaml, verbose=true)

#=
    Given a PEtab model we can create a PEtabODEProblem (in the future we plan to add surrogate, SDE, etc... problems). 
    
    There are several user options availble when creating the PEtab ODE-problem (full list in documentation), main are: 
    1) ODE-solver options - Which ODE solver and which solver tolerances (abstol and reltol). Below we use the ODE solver
       Rodas5P() (works well for smaller models ≤ 15 states), and we use the default abstol=reltol=1e-8. 
    2) Gradient option - Which gradient option to use (full list in documentation). For small models like Boehm forward 
       mode automatic differentitation (AD) is fastest, thus below we choose :ForwardDiff.
    3) Hessian option - Which gradient option to use (full list in documentation). For small models like Boehm with 
       ≤20 parameters it is computationally feasible to compute the fill Hessian via forward-mode AD. Thus below 
       we choose :ForwardDiff. 

    Note - For :ForwardDiff the user can set the Chunk-size (see https://juliadiff.org/ForwardDiff.jl/stable/). This can improve performance, 
    and we plan to add automatic tuning of it.
=#
petab_problem = PEtabODEProblem(petab_model, 
                                ode_solver=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8), 
                                gradient_method=:ForwardDiff, 
                                hessian_method=:ForwardDiff)

#=   
    ODESolverThe PEtabODEProblem has everything needed to set up an optimization problem with most availble optimizers. 
    
    The main fields are;
    1) petabODEProblem.compute_cost - Given a parameter vector θ computes the cost (objective function). 
    2) petabODEProblem.compute_gradient!- Given a parameter vector θ computes the gradient using the choosen method
    3) petabODEProblem.compute_hessian!- Given a parameter vector θ computes the gradient using the choosen method
    4) petabODEProblem.lower_bounds- A vector with the lower bounds for parameters as specified by the PEtab parameters file. 
    5) petabODEProblem.upper_bounds- A vector with the upper bounds for parameters as specified by the PEtab parameters file. 
    6) petabODEProblem.θ_names- A vector with the names of the parameters to estimate. 

    Note1 - The parameter vector θ is assumed to be on PEtab specfied parameter-scale. Thus if parameter i is on is on the 
    log-scale so should θ[i] be. 

    Note2 - The compute_gradient! and compute_hessian! functions are in-place functions. Thus their first argument is
    an already pre-allocated gradient and hessian respectively (see below)
=#
p = petab_problem.θ_nominalT # Parameter values in the PEtab file on log-scale
gradient = zeros(length(p))
hessian = zeros(length(p), length(p))
cost = petab_problem.compute_cost(p)
petab_problem.compute_gradient!(gradient, p)
petab_problem.compute_hessian!(hessian, p)
@printf("Cost for Boehm = %.2f\n", cost)
@printf("First element in the gradient for Boehm = %.2e\n", gradient[1])
@printf("First element in the hessian for Boehm = %.2f\n", hessian[1, 1])
