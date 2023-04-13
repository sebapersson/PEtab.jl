#= 
    Example for how to set a PEtab ODE problem for the small Branmark model which has a preequilibration condition.
    This means that before the main simulation where we match the model against data the model must first be 
    at a steady state; du = f(u, p, t) ≈ 0 which can be achieved via
        1) Simulations
        2) Rootfinding

    In this example we setup the PEtab problem with the best options for upto a medium sized model (≤ 75 states).
    
    Further examples include;
    Boehm.jl - here we show how to best handle small models (states ≤ 20, parameters ≤ 20). We further cover more details 
        about the important readPEtabModel and setupPEtabODEProblem functions. Recommended to checkout before looking at
        Bachmann.jl, Beer.jl and Brannmark.jl
    Bachmann.jl - here we show how to set the best options for a medium sized model (20 ≤ states ≤ 50, 20 ≤ parameters ≤ 75), 
        and how to compute gradients via adjoint sensitivity analysis.
    Beer.jl - here we show how to handle models when majority of parameter are specific to specific experimental conditions.
=#

using PEtab
using OrdinaryDiffEq
using Printf

# Create the PEtabModel 
pathYaml = joinpath(@__DIR__, "Brannmark", "Brannmark_JBC2010.yaml") # @__DIR__ = file directory
petabModel = readPEtabModel(pathYaml, verbose=true)


#=
    --- Preequilibration  ---
    For models with preequilibration we must before the main simulation solve for the steady state du = f(u, p, t) ≈ 0.
    This can be done via i) :Rootfinding where we use any algorithm from NonlinearSolve.jl to find the roots of f, 
    and by ii) :Simulate where from the initial condition using an ODE-solver we simulate the ODE-model until it 
    reaches a steady state. The second option is more stable and often performs best.

    When creating a PEtabODEProblem we can set steady-state solver options via the function getSteadyStateSolverOptions,
    where the first argument is method which can be either :Rootfinding or :Simulate (recommended). For :Simulate 
    we can choose how to terminate steady-state simulation via the howCheckSimulationReachedSteadyState argument which 
    accepts:
        1) :wrms : Weighted root-mean square √(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1
        2)-:Newton : If Newton-step Δu is sufficiently small √(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1. 
    Newton often perform better but requires an invertible Jacobian. In case not fulfilled the code switches automatically
    to wrms. (abstol, reltol) defaults to ODE solver tolerances divided by 100.

    Note1 - In case a SteadyStateSolverOption is not specified the default is :Simulate with :wrms.
    Note2 - A separate steady-state solver option can also be set for the gradient.
    Note3 - All gradient and hessian options are compatible with :Simulate. :Rootfinding is only compatible 
            with approaches using Forward-mode automatic differentiation.
=#
odeSolverOptions = getODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
ssOptions = getSteadyStateSolverOptions(:Simulate,
                                        howCheckSimulationReachedSteadyState=:wrms)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    ssSolverOptions=ssOptions,
                                    gradientMethod=:ForwardDiff) 
p = petabProblem.θ_nominalT # Parameter values in the PEtab file on log-scale
gradient = zeros(length(p)) # In-place gradients 
cost = petabProblem.computeCost(p)
petabProblem.computeGradient!(gradient, p)
@printf("Cost for Brannmark = %.2f\n", cost)
@printf("First element in the gradient for Brannmark = %.2e\n", gradient[1])
