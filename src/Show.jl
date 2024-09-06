#=
    Functions for better printing of relevant PEtab-structs which are
    exported to the user.
=#
import Base.show

function _get_solver_show(solver::ODESolver)::Tuple{String, String}
    @unpack abstol, reltol, maxiters = solver
    options = @sprintf("(abstol, reltol, maxiters) = (%.1e, %.1e, %.0e)", abstol, reltol,
                       maxiters)
    _solver = match(r"^[^({]+", solver.solver |> string).match
    return _solver, options
end

function _get_ss_solver_show(ss_solver::SteadyStateSolver; onlyheader::Bool = false)
    if ss_solver.method === :Simulate
        heading = styled"{magenta:Simulate} ODE until du = f(u, p, t) ≈ 0"
        if ss_solver.termination_check === :wrms
            opt = styled"\nTerminates when {magenta:wrms} = \
                        (∑((du ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
        else
            opt = styled"\nTerminates when {magenta:Newton} step Δu = \
                         √(∑((Δu ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
        end
    else
        heading = styled"{magenta:Rootfinding} to solve du = f(u, p, t) ≈ 0"
        alg = match(r"^[^({]+", ss_solver.rootfinding_alg |> string).match
        opt = styled"\n{magenta:Algorithm:} $(alg) with NonlinearSolve.jl termination"
    end
    if onlyheader
        return heading
    else
        return styled"$(heading)$(opt)"
    end
end

function show(io::IO, solver::ODESolver)
    _solver, options = _get_solver_show(solver)
    str = styled"{blue:{bold:ODESolver:}} {magenta:$(_solver)} with options $options"
    print(io, str)
end
function show(io::IO, ss_solver::SteadyStateSolver)
    str = styled"{blue:{bold:SteadyStateSolver:}} "
    options = _get_ss_solver_show(ss_solver)
    print(io, styled"$(str)$(options)")
end
function show(io::IO, parameter::PEtabParameter)
    header = styled"{blue:{bold:PEtabParameter:}} {magenta:$(parameter.parameter)} "
    @unpack scale, lb, ub, prior = parameter
    opt = @sprintf("estimated on %s-scale with bounds [%.1e, %.1e]", scale, lb, ub)
    if !isnothing(prior)
        prior_str = replace(prior |> string, r"\{[^}]+\}" => "")
        opt *= @sprintf(" and prior %s", prior_str)
    end
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, observable::PEtabObservable)
    @unpack obs, noise_formula, transformation = observable
    header = styled"{blue:{bold:PEtabObservable:}} "
    opt1 = styled"{magenta:h} = $(obs) and {magenta:sd} = $(noise_formula)"
    if transformation in [:log, :log10]
        opt = styled"$opt1 with log-normal measurement noise"
    else
        opt = styled"$opt1 with normal measurement noise"
    end
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, event::PEtabEvent)
    @unpack condition, target, affect = event
    header = styled"{blue:{bold:PEtabEvent:}} "
    if is_number(string(condition))
        _cond = "t == " * string(condition)
    else
        _cond = condition |> string
    end
    if target isa Vector
        target_str = "["
        for tg in target
            target_str *= (tg |> string) * ", "
        end
        target_str = target_str[1:(end - 2)] * "]"
    else
        target_str = target |> string
    end
    if affect isa Vector
        affect_str = "["
        for af in affect
            affect_str *= (af |> string) * ", "
        end
        affect_str = affect_str[1:(end - 2)] * "]"
    else
        affect_str = affect |> string
    end
    effect_str = target_str * " = " * affect_str
    opt = styled"{magenta:Condition} $_cond and {magenta:affect} $effect_str"
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, model::PEtabModel)
    nstates = @sprintf("%d", length(unknowns(model.sys_mutated)))
    nparameters = @sprintf("%d", length(parameters(model.sys_mutated)))
    header = styled"{blue:{bold:PEtabModel:}} {magenta:$(model.name)} with $nstates states \
                    and $nparameters parameters"
    if haskey(model.paths, :dirjulia)
        opt = @sprintf("\nGenerated Julia model files are at %s", model.paths[:dirjulia])
    else
        opt = ""
    end
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, prob::PEtabODEProblem)
    @unpack probinfo, model_info, nparameters_esimtate = prob
    name = model_info.model.name
    nstates = @sprintf("%d", length(unknowns(model_info.model.sys_mutated)))
    nest = @sprintf("%d", nparameters_esimtate)

    header = styled"{blue:{bold:PEtabODEProblem:}} {magenta:$(name)} with ODE-states \
                    $nstates and $nest parameters to estimate"

    optheader = styled"\n---------------- {blue:{bold:Problem options}} ---------------\n"
    opt1 = styled"Gradient method: {magenta:$(probinfo.gradient_method)}\n"
    opt2 = styled"Hessian method: {magenta:$(probinfo.hessian_method)}\n"
    solver1, options1 = _get_solver_show(probinfo.solver)
    solver2, options2 = _get_solver_show(probinfo.solver_gradient)
    opt3 = styled"ODE-solver nllh: {magenta:$(solver1)}\n"
    opt4 = styled"ODE-solver gradient: {magenta:$(solver2)}"
    if model_info.simulation_info.has_pre_equilibration == false
        print(io, styled"$(header)$(optheader)$(opt1)$(opt2)$(opt3)$(opt4)")
        return nothing
    end
    ss_solver1 = _get_ss_solver_show(probinfo.ss_solver; onlyheader = true)
    ss_solver2 = _get_ss_solver_show(probinfo.ss_solver_gradient, onlyheader = true)
    opt5 = styled"\nss-solver: $(ss_solver1)\n"
    opt6 = styled"ss-solver gradient: $(ss_solver2)"
    print(io, styled"$(header)$(optheader)$(opt1)$(opt2)$(opt3)$(opt4)$(opt5)$(opt6)")
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
function show(io::IO, target::PEtabLogDensity)
    printstyled(io, "PEtabLogDensity", color = 116)
    print(io, " with ", target.dim, " parameters to infer.")
end
