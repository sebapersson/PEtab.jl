#=
    Functions for better printing of relevant PEtab-structs which are
    exported to the user.
=#
import Base.show

StyledStrings.addface!(:PURPLE => StyledStrings.Face(foreground = 0x8f4093))

function _get_solver_show(solver::ODESolver)::Tuple{String, String}
    @unpack abstol, reltol, maxiters = solver
    options = @sprintf("abstol=%.1e, reltol=%.1e, maxiters=%.0e", abstol, reltol,
                       maxiters)
    # First needed to handle expressions on the form OrdinaryDiffEq.Rodas5P...
    _solver = string(solver.solver)
    if length(_solver) ≥ 14 && _solver[1:14] == "OrdinaryDiffEq"
        _solver = split(string(solver.solver), ".")[2:end] |> prod
    end
    _solver = match(r"^[^({]+", _solver).match
    if length(_solver) ≥ 9 && _solver[1:9] == "Sundials."
        _solver = replace(_solver, "Sundials." => "")
    end
    return _solver, options
end

function _get_ss_solver_show(ss_solver::SteadyStateSolver; onlyheader::Bool = false)
    if ss_solver.method === :Simulate
        heading = styled"{emphasis:Simulate} ODE until du = f(u, p, t) ≈ 0"
        if ss_solver.termination_check === :wrms
            opt = styled"\nTerminates when {emphasis:wrms} = \
                        (∑((du ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
        else
            opt = styled"\nTerminates when {emphasis:Newton} step Δu = \
                         √(∑((Δu ./ (reltol * u .+ abstol)).^2) / len(u)) < 1"
        end
    else
        heading = styled"{emphasis:Rootfinding} to solve du = f(u, p, t) ≈ 0"
        alg = match(r"^[^({]+", ss_solver.rootfinding_alg |> string).match
        alg = replace(alg, "NonlinearSolveFirstOrder." => "")
        opt = styled"\n{emphasis:Algorithm:} $(alg) with NonlinearSolve.jl termination"
    end
    if onlyheader
        return heading
    else
        return styled"$(heading)$(opt)"
    end
end

function show(io::IO, solver::ODESolver)
    _solver, options = _get_solver_show(solver)
    str = styled"{PURPLE:{bold:ODESolver}} {emphasis:$(_solver)}: $options"
    print(io, str)
end
function show(io::IO, ss_solver::SteadyStateSolver)
    str = styled"{PURPLE:{bold:SteadyStateSolver:}} "
    options = _get_ss_solver_show(ss_solver)
    print(io, styled"$(str)$(options)")
end
function show(io::IO, parameter::PEtabParameter)
    @unpack parameter_id, scale, estimate, prior, value, lb, ub = parameter
    header = styled"{PURPLE:{bold:PEtabParameter}} {emphasis:$(parameter_id)}: "

    if estimate == false
        opt = @sprintf("fixed = %.2e", value)
    end

    if estimate == true && isnothing(prior)
        opt = @sprintf("estimate (scale = %s, bounds = [%.1e, %.1e])", scale, lb, ub)
    end

    if estimate == true && !isnothing(prior)
        prior_str = replace(string(prior), r"\{[^}]+\}" => "")
        prior_str = replace(prior_str, "Distributions." => "")
        opt = @sprintf("estimate (scale = %s, prior(%s) = %s)", scale, parameter_id, prior_str)
    end

    print(io, styled"$(header)$(opt)")
end
function show(io::IO, observable::PEtabObservable)
    @unpack observable_formula, observable_id, noise_formula, distribution = observable
    header = styled"{PURPLE:{bold:PEtabObservable}} {emphasis:$(observable_id)}: "
    if any(occursin.(["+", "-"], observable_formula))
        observable_formula = "($(observable_formula))"
    end
    if any(occursin.(["+", "-"], noise_formula))
        noise_formula = "($(noise_formula))"
    end

    if distribution == Normal
        opt = "data ~ Normal(μ=$(observable_formula), σ=$(noise_formula))"
    elseif distribution == LogNormal
        opt = "log(data) ~ Normal(μ=log($(observable_formula)), σ=$(noise_formula))"
    elseif distribution == Log2Normal
        opt = "log2(data) ~ Normal(μ=log2($(observable_formula)), σ=$(noise_formula))"
    elseif distribution == Log10Normal
        opt = "log10(data) ~ Normal(μ=log10($(observable_formula)), σ=$(noise_formula))"
    elseif distribution == Laplace
        opt = "data ~ Laplace(μ=$(observable_formula), θ=$(noise_formula))"
    elseif distribution == LogLaplace
        opt = "log(data) ~ Laplace(μ=log($(observable_formula)), θ=$(noise_formula))"
    end
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, event::PEtabEvent)
    @unpack condition, target_ids, target_values = event
    header = styled"{PURPLE:{bold:PEtabEvent}} "
    if is_number(string(condition))
        _cond = "t == " * string(condition)
    else
        _cond = condition
    end

    assignments = ""
    for i in eachindex(target_ids)
        assignments *= "$(target_ids[i]) => $(target_values[i]), "
    end
    assignments = assignments[1:end-2]

    opt = styled"when $(_cond): $(assignments)"
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, condition::PEtabCondition)
    @unpack condition_id, target_ids, target_values = condition
    header = styled"{PURPLE:{bold:PEtabCondition}} {emphasis:$(condition_id)}:"
    if isempty(target_ids)
        return print(io, styled"$(header)")
    end

    opt = ""
    for i in eachindex(target_ids)
        opt *= "$(target_ids[i]) => $(target_values[i]), "
    end
    opt = opt[1:end-2]
    print(io, styled"$(header) $(opt)")
end
function show(io::IO, model::PEtabModel)
    header = styled"{PURPLE:{bold:PEtabModel}} $(model.name)"
    if haskey(model.paths, :dirjulia) && !isempty(model.paths[:dirjulia])
        opt = @sprintf("\nGenerated Julia model files are at %s", model.paths[:dirjulia])
    else
        opt = ""
    end
    print(io, styled"$(header)$(opt)")
end
function show(io::IO, prob::PEtabODEProblem)
    @unpack probinfo, model_info, nparameters_estimate = prob
    name = model_info.model.name
    nest = @sprintf("%d", nparameters_estimate)
    header = styled"{PURPLE:{bold:PEtabODEProblem}} {emphasis:$(name)}: $nest parameters \
        to estimate\n(for more statistics, call `describe(petab_prob)`)"
    print(io, styled"$(header)")
end

function show(io::IO, res::PEtabOptimisationResult)
    header = styled"{PURPLE:{bold:PEtabOptimisationResult}}\n"
    opt1 = @sprintf("  min(f)                = %.2e\n", res.fmin)
    opt2 = @sprintf("  Parameters estimated  = %d\n", length(res.x0))
    opt3 = @sprintf("  Optimiser iterations  = %d\n", res.niterations)
    opt4 = @sprintf("  Runtime               = %.1es\n", res.runtime)
    opt5 = @sprintf("  Optimiser algorithm   = %s\n", res.alg)
    print(io, styled"$(header)$(opt1)$(opt2)$(opt3)$(opt4)$(opt5)")
end
function show(io::IO, res::PEtabMultistartResult)
    header = styled"{PURPLE:{bold:PEtabMultistartResult}}\n"
    opt1 = @sprintf("  min(f)                = %.2e\n", res.fmin)
    opt2 = @sprintf("  Parameters estimated  = %d\n", length(res.xmin))
    opt3 = @sprintf("  Number of multistarts = %d\n", res.nmultistarts)
    opt4 = @sprintf("  Optimiser algorithm   = %s\n", res.alg)
    if !isnothing(res.dirsave)
        opt5 = @sprintf("  Results saved at %s\n", res.dirsave)
    else
        opt5 = ""
    end
    print(io, styled"$(header)$(opt1)$(opt2)$(opt3)$(opt4)$(opt5)")
end
function show(io::IO, alg::IpoptOptimizer)
    print(io, "Ipopt(LBFGS = $(alg.LBFGS))")
end
function show(io::IO, target::PEtabLogDensity)
    out = styled"{PURPLE:{bold:PEtabLogDensity}} with $(target.dim) parameters to infer"
    print(io, out)
end

"""
    describe(prob::PEtabODEProblem)

Print summary and configuration statistics for `prob`
"""
function describe(prob::PEtabODEProblem)
    print(_describe(prob))
end

function _describe(prob::PEtabODEProblem; styled::Bool = true)
    # Get problem statistics
    @unpack probinfo, model_info, nparameters_estimate = prob
    model = prob.model_info.model
    name = model_info.model.name
    nstates = @sprintf("%d", length(unknowns(model.sys_mutated)))
    nparameters = @sprintf("%d", _get_n_parameters_sys(model.sys_mutated))
    nest = @sprintf("%d", nparameters_estimate)
    n_observables = length(unique(model.petab_tables[:measurements].observableId))
    n_conditions = length(model_info.simulation_info.conditionids[:experiment])

    header = styled"{PURPLE:{bold:PEtabODEProblem}} {emphasis:$(name)}\n"
    opt_head = styled"{underline:Problem statistics}\n"
    opt1 = "  Parameters to estimate: $nest\n"
    opt2 = "  ODE: $nstates states, $nparameters parameters\n"
    opt3 = "  Observables: $(n_observables)\n"
    opt4 = "  Simulation conditions: $(n_conditions)\n"
    model_stat = styled"$(opt_head)$(opt1)$(opt2)$(opt3)$(opt4)\n"

    opt_head = styled"{underline:Configuration}\n"
    opt1 = styled"  Gradient method: $(probinfo.gradient_method)\n"
    opt2 = styled"  Hessian method: $(probinfo.hessian_method)\n"
    solver1, options1 = _get_solver_show(probinfo.solver)
    solver2, options2 = _get_solver_show(probinfo.solver_gradient)
    opt3 = styled"  ODE solver (nllh): $solver1 ($options1)\n"
    opt4 = styled"  ODE solver (grad): $solver2 ($options2)"
    if model_info.simulation_info.has_pre_equilibration == true
        ss_solver1 = _get_ss_solver_show(probinfo.ss_solver; onlyheader = true)
        ss_solver2 = _get_ss_solver_show(probinfo.ss_solver_gradient, onlyheader = true)
        opt5 = "\n  ss-solver (nllh): $(ss_solver1)\n"
        opt6 = "  ss-solver (grad): $(ss_solver2)\n"
    else
        opt5 = ""
        opt6 = ""
    end
    comp_stat = styled"$(opt_head)$(opt1)$(opt2)$(opt3)$(opt4)$(opt5)$(opt6)"
    if styled
        return styled"$(header)$(model_stat)$(comp_stat)"
    else
        return "$(header)$(model_stat)$(comp_stat)"
    end
end
