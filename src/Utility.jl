"""
    get_odesol(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult, Vector{Float64}},
               petab_problem::PEtabODEProblem;
               condition_id::Union{String, Symbol, Nothing}=nothing,
               pre_eq_id::Union{String, Symbol, Nothing}=nothing)

From a fitted PEtab model or parameter vector retrieve the ODE solution for the specified `condition_id`.

If `condition_id` is provided, the parameters are extracted for that specific simulation condition. If not provided,
parameters for the first (default) simulation condition are returned.

If a `pre_eq_id` is provided, the initial values are taken from the pre-equilibration simulation corresponding to
`pre_eq_id`.

If a parameter vector is provided it must have the parameters in the same order as `petab_problem.xnames`.

Potential events are accounted for when solving the ODE model. The ODE solver options specified when creating the
`petab_problem` are used for solving the model.
"""
function get_odesol(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult,
                               Vector{Float64}},
                    petab_problem::PEtabODEProblem;
                    condition_id::Union{String, Symbol, Nothing} = nothing,
                    pre_eq_id::Union{String, Symbol, Nothing} = nothing)
    ode_solver = petab_problem.probinfo.solver
    simulation_info = petab_problem.model_info.simulation_info
    if isnothing(condition_id)
        condition_id = simulation_info.conditionids[:simulation][1]
    end
    if condition_id isa String
        condition_id = Symbol(condition_id)
    end
    if !isnothing(pre_eq_id) && pre_eq_id isa String
        pre_eq_id = Symbol(pre_eq_id)
    end
    if !isnothing(pre_eq_id)
        tmax_id = Symbol(string(pre_eq_id) * string(condition_id))
    else
        tmax_id = condition_id
    end

    ode_problem, cbset = get_odeproblem(res, petab_problem; condition_id = condition_id,
                                         pre_eq_id = pre_eq_id)
    @unpack solver, abstol, reltol = ode_solver
    return solve(ode_problem, solver, abstol = abstol, reltol = reltol, callback = cbset)
end

"""
    get_odeproblem(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult, Vector{Float64}},
                   petab_problem::PEtabODEProblem;
                   condition_id::Union{String, Symbol, Nothing}=nothing,
                   pre_eq_id::Union{String, Symbol, Nothing}=nothing)

From a fitted PEtab model or parameter vector retrieve the `ODEProblem` and callbacks to simulate the model for the specified `condition_id`.

If `condition_id` is provided, the parameters are extracted for that specific simulation condition. If not provided,
parameters for the first (default) simulation condition are returned.

If a `pre_eq_id` is provided, the initial values are taken from the pre-equilibration simulation corresponding to
`pre_eq_id`.

If a parameter vector is provided it must have the parameters in the same order as `petab_problem.xnames`.

Potential events are returned as second argument, and potential time of events (`tstops`) are returned as
third argument.

## Example
```julia
using OrdinaryDiffEq
# Solve the model with callbacks
prob, cb, tstops = get_odeproblem(res, petab_problem, condition_id="cond1")
sol = solve(prob, Rodas5P(), callback=cb, tstops=tstops)
```
"""
function get_odeproblem(res::Union{PEtabOptimisationResult,
                                   PEtabMultistartOptimisationResult, Vector{Float64}},
                        petab_problem::PEtabODEProblem;
                        condition_id::Union{String, Symbol, Nothing} = nothing,
                        pre_eq_id::Union{String, Symbol, Nothing} = nothing)
    @unpack simulation_info, model = petab_problem.model_info
    if isnothing(condition_id)
        condition_id = simulation_info.conditionids[:simulation][1]
    end
    u0, p = _get_fitted_parameters(res, petab_problem, condition_id, pre_eq_id, true)
    tmax = simulation_info.tmaxs[condition_id]
    ode_problem = ODEProblem(model.sys, u0, [0.0, tmax], p, jac = true)
    ode_problem = remake(ode_problem, p = ode_problem.p.tunable)
    cbset = model.callbacks
    return ode_problem, cbset
end

"""
    get_ps(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult, Vector{Float64}},
           petab_problem::PEtabODEProblem;
           condition_id::Union{String, Symbol, Nothing}=nothing,
           retmap::Bool=true)

From a fitted PEtab model or parameter vector retrieve the ODE parameters to simulate the model for the specified `condition_id`.

If `condition_id` is provided, the parameters are extracted for that specific simulation condition. If not provided,
parameters for the first (default) simulation condition are returned.

If a parameter vector is provided it must have the parameters in the same order as `petab_problem.xnames`.

If `retmap=true`, a parameter vector is returned; otherwise, a vector is returned.
"""
function get_ps(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult,
                           Vector{Float64}},
                petab_problem::PEtabODEProblem;
                condition_id::Union{String, Symbol, Nothing} = nothing,
                retmap = true)
    u0, p = _get_fitted_parameters(res, petab_problem, condition_id, nothing, retmap)
    return p
end

"""
    get_u0(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult, Vector{Float64}},
           petab_problem::PEtabODEProblem;
           condition_id::Union{String, Symbol}=nothing,
           pre_eq_id::Union{String, Symbol, Nothing}=nothing,
           retmap::Bool=true)

From a fitted PEtab model or parameter vector retrieve the inital values (u0) to simulate the model for the specified `condition_id`.

If `condition_id` is provided, the initial values are extracted for that specific simulation condition. If not provided,
initial values for the first (default) simulation condition are returned.

If a `pre_eq_id` is provided, the initial values are taken from the pre-equilibration simulation corresponding to
`pre_eq_id`. If there are potential overrides of initial values in the simulation conditions, they take priority over
the pre-equilibrium simulation.

If a parameter vector is provided it must have the parameters in the same order as `petab_problem.xnames`.

If `retmap=true`, a parameter vector is returned; otherwise, a vector is returned.
"""
function get_u0(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult,
                           Vector{Float64}},
                petab_problem::PEtabODEProblem;
                condition_id::Union{String, Symbol, Nothing} = nothing,
                pre_eq_id::Union{String, Symbol, Nothing} = nothing,
                retmap::Bool = true)
    u0, p = _get_fitted_parameters(res, petab_problem, condition_id, pre_eq_id, retmap)
    return u0
end

function _get_fitted_parameters(res::Union{PEtabOptimisationResult,
                                           PEtabMultistartOptimisationResult,
                                           Vector{Float64}},
                                petab_problem::PEtabODEProblem,
                                condition_id::Union{String, Symbol, Nothing},
                                pre_eq_id::Union{String, Symbol, Nothing},
                                retmap::Bool = true)
    ode_problem = petab_problem.probinfo.odeproblem
    cache = petab_problem.probinfo.cache
    model_info = petab_problem.model_info
    @unpack θ_indices, model, simulation_info = petab_problem.model_info
    # Sanity check input
    if isnothing(condition_id)
        _c_id = simulation_info.conditionids[:simulation][1]
    else
        _c_id = Symbol(condition_id)
        @assert _c_id ∈ simulation_info.conditionids[:simulation] "A simulation condition id was given that could not be found in among the petab model conditions."
    end
    if !isnothing(pre_eq_id)
        _pre_eq_id = Symbol(pre_eq_id)
        @assert _pre_eq_id ∈ simulation_info.conditionids[:pre_equilibration] "A pre-equilbration simulation condition id was given that could not be found in among the petab model conditions."
    else
        _pre_eq_id = nothing
    end

    p, ps = ode_problem.p[:], θ_indices.xids[:sys]
    u0, u0s = ode_problem.u0[:], first.(model.statemap)

    if res isa Vector{Float64}
        θT = transform_x(res, θ_indices.xids[:estimate], θ_indices)
    else
        θT = transform_x(res.xmin, θ_indices.xids[:estimate], θ_indices)
    end
    xdynamic, xobservable, xnoise, xnondynamic = split_x(θT, θ_indices)

    # Set constant model parameters
    _set_cond_const_parameters!(p, xdynamic, θ_indices)
    ode_problem.p .= p
    ode_problem.u0 .= u0

    # In case of no pre-eq condition we are done after changing to the condition s
    # Condition specific parameters
    if isnothing(pre_eq_id)
        oprob = _switch_condition(ode_problem, _c_id, xdynamic, model_info, cache)
        u0 = oprob.u0 |> deepcopy
        p = oprob.p |> deepcopy
        _u0 = retmap ? Pair.(u0s, u0) : u0
        _p = retmap ? Pair.(ps, p) : p
        ip = findall(x -> !occursin("__init__", x), string.(ps))
        return _u0, _p[ip]
    end

    # For models with pre-eq in order to correctly return the initial values the model
    # must first be simulated to steady state, and following the parameters must
    # be set correctly
    u_ss = Vector{Float64}(undef, length(u0))
    u_t0 = Vector{Float64}(undef, length(u0))
    change_simulation_condition! = (p_ode_problem, u0, conditionId) -> _change_simulation_condition!(p_ode_problem,
                                                                                                     u0,
                                                                                                     conditionId,
                                                                                                     xdynamic,
                                                                                                     model,
                                                                                                     θ_indices)

    pre_eq_sol = solve_ode_pre_equlibrium!(u_ss,
                                           u_t0,
                                           remake(ode_problem, p = p, u0 = u0),
                                           change_simulation_condition!,
                                           _pre_eq_id,
                                           petab_problem.ode_solver,
                                           petab_problem.ss_solver,
                                           model.convert_tspan)

    change_simulation_condition!(p, u0, _c_id)
    # Sometimes the experimentaCondition-file changes the initial values for a state
    # whose value was changed in the preequilibration-simulation. The experimentaCondition
    # value is prioritized by only changing u0 to the steady state value for those states
    # that were not affected by change to shift_expid.
    has_not_changed = u0 .== u_t0
    u0[has_not_changed] .= u_ss[has_not_changed]

    # According to the PEtab standard we can sometimes have that initial assignment is overridden for
    # pre-eq simulation, but we do not want to override for main simulation which is done automatically
    # by change_simulation_condition!. These cases are marked as NaN
    u0[isnan.(ode_problem.u0)] .= u_ss[isnan.(u0)]

    # Filter out any potential __init__ parameters as ps is returned for the non mutated system, specifically
    # to easily compute Jacobians if the intial value for a specie is set as a simulation condition we mutate
    # the system to easily compute Jacobians
    ip = findall(x -> !occursin("__init__", x), string.(ps))

    _u0 = retmap ? Pair.(u0s, u0) : u0
    _p = retmap ? Pair.(ps, p) : p
    return _u0, _p[ip]
end

"""
    solve_all_conditions(xpetab, petab_problem::PEtabODEProblem, solver; <keyword arguments>)

Simulates the ODE model for all simulation conditions using the provided ODE solver and parameter vector `xpetab`.

The parameter vector `xpetab` should be provided on the PEtab scale (default log10).

# Keyword Arguments
- `abstol=1e-8`: Absolute tolerance for the ODE solver.
- `reltol=1e-8`: Relative tolerance for the ODE solver.
- `maxiters=1e4`: Maximum iterations for the ODE solver.
- `ntimepoints_save=0`: Specifies the number of time points at which to save the ODE
    solution for each condition. A value of 0 means the solution is saved at the solvers
    default time points.
- `save_observed_t=false`: When set to true, this option overrides `ntimepoints_save`
    and saves the ODE solution only at the time points where measurement data are available.

# Returns
- `odesols`: A dictionary containing the `ODESolution` for each condition.
- `could_solve`: A boolean value indicating whether the model was successfully solved for
    all conditions.
"""
function solve_all_conditions(xpetab, petab_problem::PEtabODEProblem, solver; abstol = 1e-8,
                              reltol = 1e-8, maxiters = nothing, ntimepoints_save = 0,
                              save_observed_t = false)
    @unpack ode_problem, model, simulation_info, θ_indices = petab_problem
    @unpack ode_solver, ss_solver, cache = petab_problem
    _ode_solver = deepcopy(ode_solver)
    _ode_solver.abstol = abstol
    _ode_solver.reltol = reltol
    _ode_solver.solver = solver
    if !isnothing(maxiters)
        _ode_solver.maxiters = maxiters
    end

    xdynamic, xobservable, xnoise, xnondynamic = split_x(xpetab, θ_indices)
    xdynamic_ps = transform_x(xdynamic, θ_indices.xids[:dynamic], θ_indices,
                            :xdynamic, petab_problem.cache)

    odesols, could_solve = solve_ODE_all_conditions(ode_problem, model, xdynamic_ps,
                                                    cache, simulation_info,
                                                    θ_indices, _ode_solver, ss_solver;
                                                    save_observed_t = save_observed_t,
                                                    ntimepoints_save = ntimepoints_save)
    return odesols, could_solve
end

"""
    compute_runtime_accuracy(xpetab, petab_problem, solver; <keyword arguments>)

Get runtime and accuracy for an ODE solver when simulating a model across all simulation conditions with parameter vector `xpetab`.

The parameter vector `xpetab` should be provided on the PEtab scale (default log10).

# Keyword Arguments
- `abstol=1e-8`: Absolute tolerance for the ODE solver.
- `reltol=1e-8`: Relative tolerance for the ODE solver.
- `solver_high_acc=Rodas4P()`: The ODE solver used to generate a high accuracy solution,
    which is used as reference when computing the high accuracy soluation.
- `abstol_highacc=1e-12`: Absolute tolerance for the high accuracy ODE solver.
- `reltol_highacc=1e-12`: Relative tolerance for the high accuracy ODE solver.
- `compute_acc=true`: If set to `false`, accuracy is not evaluated (returned a 0).
- `ntimes_solve=5`: Number times to simulated the model to determine the average runtime.

# Returns
- `runtime`: The average time taken to solve the model across all conditions, in seconds.
- `acc`: The solver's accuracy, determined by comparison with the high accuracy ODE solver.
"""
function compute_runtime_accuracy(xpetab, petab_problem, solver; abstol = 1e-8,
                                  reltol = 1e-8, solver_high_acc = Rodas4P(),
                                  abstol_highacc = 1e-12, reltol_highacc = 1e-12,
                                  compute_acc::Bool = true, ntimes_solve = 5)
    local sols_highacc, could_solve_highacc
    if compute_acc == true
        sols_highacc, could_solve_highacc = PEtab.solve_all_conditions(xpetab,
                                                                       petab_problem,
                                                                       solver_high_acc;
                                                                       abstol = abstol_highacc,
                                                                       reltol = reltol_highacc,
                                                                       ntimepoints_save = 100)
        if could_solve_highacc == false
            @error "Could not solve high accuracy solution. Consider changing solver_high_acc"
        end
    else
        sols_highacc, could_solve_highacc = nothing, nothing
    end

    # Get accuracy
    if !isnothing(sols_highacc)
        sols, could_solve = PEtab.solve_all_conditions(xpetab, petab_problem, solver;
                                                       abstol = abstol, reltol = reltol,
                                                       ntimepoints_save = 100)
        if could_solve == true
            acc = 0.0
            for id in keys(sols)
                acc += sum((Array(sols[id]) - Array(sols_highacc[id])) .^ 2)
            end
        end
    else
        # Check if we can solve the ODE
        acc = nothing
        _, could_solve = PEtab.solve_all_conditions(xpetab, petab_problem, solver;
                                                    abstol = abstol, reltol = reltol,
                                                    ntimepoints_save = 100)
    end

    if could_solve == false
        @warn "Could not solve ODE. Runtime and accuracy are returned as Inf" maxlog=10
        return Inf, Inf
    end

    # Get runtime
    runtime = 0.0
    for i in 1:ntimes_solve
        runtime += @elapsed _, _ = PEtab.solve_all_conditions(xpetab, petab_problem, solver;
                                                              abstol = abstol,
                                                              reltol = reltol)
    end
    runtime /= ntimes_solve

    return runtime, acc
end
