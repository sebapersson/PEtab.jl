#=
    Functions for better printing of relevant PEtab-structs which are
    exported to the user.
=#
import Base.show

# Helper function for printing ODE-solver options
function get_ode_solver_str(a::ODESolver)
    solver_str = string(a.solver)[1:(findfirst('(', string(a.solver)) - 1)]
    options_str = @sprintf("(abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", a.abstol,
                           a.reltol, a.maxiters)
    return solver_str, options_str
end

function show(io::IO, a::PEtabModel)
    model_name = @sprintf("%s", a.modelname)
    n_odes = @sprintf("%d", length(states(a.sys_mutated)))
    n_ode_parameters = @sprintf("%d", length(parameters(a.sys_mutated)))

    printstyled(io, "PEtabModel", color = 116)
    print(io, " for model ")
    printstyled(io, model_name, color = 116)
    print(io, ". ODE-system has ")
    printstyled(io, n_odes * " states", color = 116)
    print(io, " and ")
    printstyled(io, n_ode_parameters * " parameters.", color = 116)
    if !isempty(a.paths[:dirjulia])
        @printf(io, "\nGenerated Julia files are at %s", a.paths[:dirjulia])
    end
end
function show(io::IO, a::ODESolver)
    # Extract ODE solver as a readable string (without everything between)
    solver_str, options_str = get_ode_solver_str(a)
    printstyled(io, "ODESolver", color = 116)
    print(io, " with ODE solver ")
    printstyled(io, solver_str, color = 116)
    @printf(io, ". Options %s", options_str)
end
function show(io::IO, prob::PEtabODEProblem)
    @unpack probleminfo, model_info = prob
    model = model_info.petab_model
    model_name = model.modelname

    n_odes = length(states(model.sys_mutated))
    n_parameters_est = length(prob.xnames)
    n_dynamic_parameters = length(model_info.θ_indices.xids[:dynamic])

    solver_str, options_str = get_ode_solver_str(probleminfo.solver)
    solver_gradient_str, options_gradient_str = get_ode_solver_str(probleminfo.solver_gradient)

    gradient_method = string(probleminfo.gradient_method)
    hessian_method = string(probleminfo.hessian_method)

    printstyled(io, "PEtabODEProblem", color = 116)
    print(io, " for ")
    printstyled(io, model_name, color = 116)
    @printf(io,
            ". ODE-states: %d. Parameters to estimate: %d where %d are dynamic.\n---------- Problem settings ----------\nGradient method : ",
            n_odes, n_parameters_est, n_dynamic_parameters)
    printstyled(io, gradient_method, color = 116)
    if !isnothing(hessian_method)
        print(io, "\nHessian method : ")
        printstyled(io, hessian_method, color = 116)
    end
    print(io, "\n--------- ODE-solver settings --------")
    printstyled(io, "\nCost ")
    printstyled(io, solver_str, color = 116)
    @printf(io, ". Options %s", options_str)
    printstyled(io, "\nGradient ")
    printstyled(io, solver_gradient_str, color = 116)
    @printf(io, ". Options %s", options_gradient_str)

    if model_info.simulation_info.has_pre_equilibration == true
        print(io, "\n--------- SS solver settings ---------")
        # Print cost steady state solver
        print(io, "\nCost ")
        printstyled(io, string(a.ss_solver.method), color = 116)
        if probleminfo.ss_solver.method === :Simulate &&
           probleminfo.ss_solver.check_simulation_steady_state === :wrms
            @printf(io, ". Option wrms with (abstol, reltol) = (%.1e, %.1e)",
                    probleminfo.ss_solver.abstol, probleminfo.ss_solver.reltol)
        elseif probleminfo.ss_solver.method === :Simulate &&
               probleminfo.ss_solver.check_simulation_steady_state === :Newton
            @printf(io, ". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)",
                    probleminfo.ss_solver.abstol, probleminfo.ss_solver.reltol)
        elseif probleminfo.ss_solver.method === :Rootfinding
            algStr = string(a.ss_solver.rootfindingAlgorithm)
            i_end = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:(i_end - 1)] * "()"
            @printf(io,
                    ". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)",
                    algStr, probleminfo.ss_solver.abstol, probleminfo.ss_solver.reltol, probleminfo.ss_solver.maxiters)
        end

        # Print gradient steady state solver
        print(io, "\nGradient ")
        printstyled(io, string(probleminfo.ss_solver_gradient.method), color = 116)
        if probleminfo.ss_solver_gradient.method === :Simulate &&
           probleminfo.ss_solver_gradient.check_simulation_steady_state === :wrms
            @printf(io, ". Options wrms with (abstol, reltol) = (%.1e, %.1e)",
                    probleminfo.ss_solver_gradient.abstol, probleminfo.ss_solver_gradient.reltol)
        elseif probleminfo.ss_solver_gradient.method === :Simulate &&
               probleminfo.ss_solver_gradient.check_simulation_steady_state === :Newton
            @printf(io, ". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)",
                    probleminfo.ss_solver_gradient.abstol, probleminfo.ss_solver_gradient.reltol)
        elseif probleminfo.ss_solver_gradient.method === :Rootfinding
            algStr = string(probleminfo.ss_solver_gradient.rootfindingAlgorithm)
            i_end = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:(i_end - 1)] * "()"
            @printf(io,
                    ". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)",
                    algStr, probleminfo.ss_solver_gradient.abstol, probleminfo.ss_solver_gradient.reltol,
                    probleminfo.ss_solver_gradient.maxiters)
        end
    end
end
function show(io::IO, a::SteadyStateSolver)
    printstyled(io, "SteadyStateSolver", color = 116)
    if a.method === :Simulate
        print(io, " with method ")
        printstyled(io, ":Simulate", color = 116)
        print(io, " ODE-model until du = f(u, p, t) ≈ 0.")
        if a.check_simulation_steady_state === :wrms
            @printf(io,
                    "\nSimulation terminated if wrms fulfill;\n√(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        else
            @printf(io,
                    "\nSimulation terminated if Newton-step Δu fulfill;\n√(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        end
    end

    if a.method === :Rootfinding
        print(io, " with method ")
        printstyled(io, ":Rootfinding", color = 116)
        print(io, " to solve du = f(u, p, t) ≈ 0.")
        if isnothing(a.rootfinding_alg)
            @printf(io, "\nAlgorithm : NonlinearSolve's heruistic. Options ")
        else
            algStr = string(a.rootfinding_alg)
            i_end = findfirst(x -> x == '(', algStr)
            algStr = algStr[1:(i_end - 1)] * "()"
            @printf(io, "\nAlgorithm : %s. Options ", algStr)
        end
    end
    if isnothing(a.abstol)
        @printf(io, "with (abstol, reltol) = default values.")
    else
        @printf(io, "with (abstol, reltol) = (%.1e, %.1e)", a.abstol, a.reltol)
    end
end
function show(io::IO, a::PEtabOptimisationResult)
    printstyled(io, "PEtabOptimisationResult", color = 116)
    print(io, "\n--------- Summary ---------\n")
    @printf(io, "min(f)                = %.2e\n", a.fmin)
    @printf(io, "Parameters esimtated  = %d\n", length(a.x0))
    @printf(io, "Optimiser iterations  = %d\n", a.n_iterations)
    @printf(io, "Run time              = %.1es\n", a.runtime)
    @printf(io, "Optimiser algorithm   = %s\n", a.alg)
end
function show(io::IO, a::PEtabMultistartOptimisationResult)
    printstyled(io, "PEtabMultistartOptimisationResult", color = 116)
    print(io, "\n--------- Summary ---------\n")
    @printf(io, "min(f)                = %.2e\n", a.fmin)
    @printf(io, "Parameters esimtated  = %d\n", length(a.xmin))
    @printf(io, "Number of multistarts = %d\n", a.n_multistarts)
    @printf(io, "Optimiser algorithm   = %s\n", a.alg)
    if !isnothing(a.dir_save)
        @printf(io, "Results saved at %s\n", a.dir_save)
    end
end
function show(io::IO, a::PEtabParameter)
    printstyled(io, "PEtabParameter", color = 116)
    @printf(io, " %s", a.parameter)
    if a.estimate != true
        return nothing
    end
    @printf(io, ". Estimated on %s-scale with bounds [%.1e, %.1e]", a.scale, a.lb, a.ub)
    if isnothing(a.prior)
        return nothing
    end
    @printf(io, " and prior %s", string(a.prior))
    return nothing
end
function show(io::IO, a::PEtabObservable)
    printstyled(io, "PEtabObservable", color = 116)
    @printf(io, ": h = %s, noise-formula = %s and ", string(a.obs), string(a.noise_formula))
    if a.transformation ∈ [:log, :log10]
        @printf(io, "log-normal measurement noise")
    else
        @printf(io, "normal (Gaussian) measurement noise")
    end
end
function show(io::IO, a::PEtabEvent)
    printstyled(io, "PEtabEvent", color = 116)
    condition = is_number(string(a.condition)) ? "t == " * string(a.condition) :
                string(a.condition)
    @printf(io, ": condition %s, affect %s = %s", condition, string(a.target),
            string(a.affect))
end
function show(io::IO, target::PEtabLogDensity)
    printstyled(io, "PEtabLogDensity", color = 116)
    print(io, " with ", target.dim, " parameters to infer.")
end
