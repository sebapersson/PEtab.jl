#= 
    Example for how to set a PEtab ODE problem for the small Boehm model and how given a parameter vector 
    performed Hessian based optmization using;
        1) Optim.jl (Interior point Newton)
        2) Ipopt.jl (Interior point Newton like method)

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
    To save time the fucntion readPEtabModel has the default forceBuildJlFiles=false meaning that the Julia files 
    are not rebuilt in case they already exist.
=#    
pathYaml = joinpath(@__DIR__, "Boehm", "Boehm_JProteomeRes2014.yaml") # @__DIR__ = file directory
petabModel = readPEtabModel(pathYaml, verbose=true)

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
odeSolverOptions = ODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
petabProblem = createPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:ForwardDiff)

#=   
    ODESolverOptionsThe PEtabODEProblem has everything needed to set up an optimization problem with most availble optimizers. 
    
    The main fields are;
    1) petabODEProblem.computeCost - Given a parameter vector θ computes the cost (objective function). 
    2) petabODEProblem.computeGradient!- Given a parameter vector θ computes the gradient using the choosen method
    3) petabODEProblem.computeHessian!- Given a parameter vector θ computes the gradient using the choosen method
    4) petabODEProblem.lowerBounds- A vector with the lower bounds for parameters as specified by the PEtab parameters file. 
    5) petabODEProblem.upperBounds- A vector with the upper bounds for parameters as specified by the PEtab parameters file. 
    6) petabODEProblem.θ_estNames- A vector with the names of the parameters to estimate. 

    Note1 - The parameter vector θ is assumed to be on PEtab specfied parameter-scale. Thus if parameter i is on is on the 
    log-scale so should θ[i] be. 

    Note2 - The computeGradient! and computeHessian! functions are in-place functions. Thus their first argument is
    an already pre-allocated gradient and hessian respectively (see below)
=#
p = petabProblem.θ_nominalT # Parameter values in the PEtab file on log-scale
gradient = zeros(length(p))
hessian = zeros(length(p), length(p))
cost = petabProblem.computeCost(p)
petabProblem.computeGradient!(gradient, p)
petabProblem.computeHessian!(hessian, p)
@printf("Cost for Boehm = %.2f\n", cost)
@printf("First element in the gradient for Boehm = %.2e\n", gradient[1])
@printf("First element in the hessian for Boehm = %.2f\n", hessian[1, 1])


#=   
    The PEtabODEProblem has as mentioned everything needed to set up an optimization problem. Below we show how to set 
    up a problem using the Interior point Newton method in Optim.jl (https://github.com/JuliaNLSolvers/Optim.jl). Extensive 
    benchmarks show that this method performs great in case we can compute the full Hessian via automatic differentitation.

    Note - the PEtabODEProblem compute gradient and hessian are on the format which Optim.jl accepts. 
=#
using Optim
using Random
# Setup the problem on Optim.jl format
nParameters = length(petabProblem.lowerBounds)
df = TwiceDifferentiable(petabProblem.computeCost, petabProblem.computeGradient!, petabProblem.computeHessian!, zeros(nParameters))
dfc = TwiceDifferentiableConstraints(petabProblem.lowerBounds, petabProblem.upperBounds)

# Generate a random parameter vector within the parameter bounds
Random.seed!(123)
p0 = [rand() * (petabProblem.upperBounds[i] - petabProblem.lowerBounds[i]) + petabProblem.lowerBounds[i] for i in eachindex(petabProblem.lowerBounds)]
res = Optim.optimize(df, dfc, p0, IPNewton(), Optim.Options(iterations = 1000, show_trace = true))
  

#=   
    Another popular optimizer is Ipopt (https://coin-or.github.io/Ipopt/). 
    
    Setting up Ipopt in Julia is a bit involved, and for anyone interested in the details we provide an Ipopt wrapper function in the file 
    Set_up_Ipopt.jl.

    Note - Ipopt.jl allows the hessian to be approximated via a L-BFGS function. The user can choose this option 
    below by specifaying hessianMethod=:LBFGS, while hessianMethod=:userProvided means that the hessian in the 
    PEtabODEProblem is used.
=#
using Ipopt
include(joinpath(@__DIR__, "Set_up_ipopt.jl"))
ipoptProblem, iterations = createIpoptProblem(petabProblem, hessianMethod=:userProvided)
ipoptProblem.x .= p0 # Set initial values
# Options 
Ipopt.AddIpoptIntOption(ipoptProblem, "print_level", 5)
Ipopt.AddIpoptIntOption(ipoptProblem, "max_iter", 1000)
sol_opt = Ipopt.IpoptSolve(ipoptProblem)
