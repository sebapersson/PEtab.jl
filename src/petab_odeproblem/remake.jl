"""
    remake(prob::PEtabODEProblem; conditions=Symbol[], parameters=nothing) -> PEtabODEProblem

Create a new `PEtabODEProblem` from `prob`, restricted to a subset of simulation conditions
and/or with a subset of parameters to estimate in `prob` fixed to constant values.

Intended for efficient subsetting (e.g. evaluating `nllh`/`grad!`/`hess!` on a subset of
conditions, or with a reduced set of parameters to estimate). Typically faster than
constructing a new `PEtabODEProblem`, since compiled functions from `prob` are reused
(avoids recompilation).

# Keyword arguments
- `conditions`: Simulation conditions to keep. If empty (default), all conditions are kept.
  Format depends on whether the model has pre-equilibration:
  - No pre-equilibration: Provide `Vector{Symbol}` of simulation condition ids (e.g.
    `[:cond1, :cond2]`).
  - With pre-equilibration: Provide `Vector{Pair}` of `pre_eq_id => simulation_id`
    (e.g. `[:pre1 => :cond1, :pre1 => :cond2]`).
- `experiments`: Experimental time course ids to keep, as Vector{`Symbol`}. Only applicable
  for problems in PEtab v2 standard format.
- `parameters`: Parameters to fix to constant values, as a vector of pairs
  `[:p1 => val1, :p2 => val2, ...]`. Only parameters that are estimated in `prob` can be
  fixed. Values are given on the **linear** scale; e.g. if a parameter is estimated on
  `:log10`, pass `val` (not `log10(val)`).

## Examples
```julia
# Keep only simulation conditions :cond1 and :cond3
prob_sub = remake(prob; conditions = [:cond1, :cond3])
```
```julia
# Fix parameters k1 and k2
prob_sub = remake(prob; parameters = [:k1 => 3.0, :k2 => 4.0])
```
"""
function SciMLBase.remake(
        @nospecialize(prob::PEtabODEProblem);
        conditions::Union{Vector{<:Pair}, Vector{Symbol}} = Symbol[],
        experiments = Symbol[],
        parameters::Vector{<:Pair{Symbol, <:Real}} = Pair{Symbol, Real}[]
    )::PEtabODEProblem
    if isempty(conditions) && isempty(parameters) && isempty(experiments)
        # Nothing to subset, but an independent copy of prob is still returned
        return _remake_condition_ids(prob, Int64[])
    end

    petab_version = _get_version(prob.model_info)
    if petab_version == "2.0.0" && !isempty(conditions)
        throw(ArgumentError("For PEtab v2 problems the `conditions` keyword is not \
            supported for `remake`; use `experiments` to subset experimental time-courses."))
    end
    if petab_version == "1.0.0" && !isempty(experiments)
        throw(ArgumentError("For PEtab v1 problems or problems defined in Julia, the \
            `experiments` keyword is not supported for `remake`; use `conditions` to \
            subset simulation conditions"))
    end

    if !isempty(experiments)
        prob = _remake_experiments(prob, experiments)
    end
    if !isempty(conditions)
        prob = _remake_conditions(prob, conditions)
    end
    if !isempty(parameters)
        prob = _remake_parameters(prob, parameters)
    end
    return prob
end

# TODO: For ML parameters? Disallow
function _remake_parameters(
        @nospecialize(prob::PEtabODEProblem), parameters::Vector{<:Pair{Symbol, <:Real}}
    )::PEtabODEProblem
    # It only makes sense to remake (from compilation point if view) if parameters that
    # before were to be estimated are set to fixated.
    for (parameter_id, _) in parameters
        if !in(parameter_id, prob.xnames)
            throw(PEtabInputError("Parameter '$(parameter_id)' is not marked as estimated \
                in the provided PEtabODEProblem and cannot be fixed via `remake`."))
        end
        if parameter_id in prob.model_info.xindices.ids[:ml_est]
            throw(PEtabInputError("Remake with respect to ML parameters is not currently \
                supported. It will be more computationally efficient (including \
                compilation time) to create a new problem than to fixate ML parameters."))
        end
    end

    # Map the fixated values to the parameter scale (they are assumed to be on the linear)
    # scale (e.g. not log10 which might be needed by the PEtab-problem)
    @unpack model_info, probinfo = prob
    x_fixed = zeros(Float64, length(parameters))
    for (i, parameter_id) in pairs(first.(parameters))
        scale = model_info.xindices.xscale[parameter_id]
        x_fixed[i] = transform_x(parameters[i].second, scale; to_xscale = true)
    end

    # Updated struct fields for the new problem
    ix = findall(x -> !(x in first.(parameters)), prob.xnames)
    xnames_new = propertynames(prob.xnominal)[ix]
    xnames_ps_new = propertynames(prob.xnominal_transformed)[ix]
    lb_new = _to_component_array(prob.lower_bounds[xnames_ps_new])
    ub_new = _to_component_array(prob.upper_bounds[xnames_ps_new])
    xnominal_new = _to_component_array(prob.xnominal[xnames_new])
    xnominal_transformed_new = _to_component_array(prob.xnominal_transformed[xnames_ps_new])
    nestimate_new = length(xnames_new)
    # Ensure xnames is of correct type; Vector{Symbol}
    xnames_new = isempty(xnames_new) ? Symbol[] : [xnames_new...]
    xnames_ps_new = isempty(xnames_ps_new) ? Symbol[] : [xnames_ps_new...]

    # Needed for the new problem (as under the hood we still use the full Hessian and
    # gradient, so these need to be pre-allocated)
    _xest_full = similar(prob.xnominal) |> collect
    _grad_full = similar(prob.xnominal) |> collect
    _H_full = zeros(Float64, length(_xest_full), length(_xest_full))
    _FIM_full = zeros(Float64, length(_xest_full), length(_xest_full))
    ix_fixed = [findfirst(x -> x == id, prob.xnames) for id in first.(parameters)]
    imap = [findfirst(x -> x == xnames_new[i], prob.xnames) for i in eachindex(xnames_new)]

    # Functions of prob are accessed via RemakeSource, so that the functions below (and
    # thus the remade problem) get the same type regardless of which problem remake was
    # called on, see the RemakeSource docstring
    src = RemakeSource(
        prob.prior, prob.grad_prior, prob.hess_prior, prob.nllh, prob.simulated_values,
        prob.chi2, prob.residuals, prob.grad!, prob.nllh_grad, prob.hess!, prob.FIM!
    )

    # PEtabODEProblem functions
    _prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return src.prior(xest_full)
    end
    _grad_prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        g = src.grad_prior(xest_full)
        return g[imap]
    end
    _hess_prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        _H = src.hess_prior(xest_full)
        H = zeros(eltype(_H), length(x), length(x))
        _map_matrix!(H, _H, imap)
        return H
    end
    _nllh = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return src.nllh(xest_full)
    end
    _simulated_values = (x; as_array = false) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return src.simulated_values(xest_full)
    end
    _chi2 = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return src.chi2(xest_full)
    end
    _residuals = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return src.residuals(xest_full)
    end
    _grad! = (g, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        src.grad!(_grad_full, xest_full)
        g .= _grad_full[imap]
        return nothing
    end
    _grad = (x) -> begin
        g = similar(x)
        _grad!(g, x)
        return g
    end
    _nllh_grad = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        nllh, _grad_full = src.nllh_grad(xest_full)
        return nllh, _grad_full[imap]
    end
    _hess! = (H, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        src.hess!(_H_full, xest_full)
        _map_matrix!(H, _H_full, imap)
        return nothing
    end
    _hess = (x) -> begin
        H = zeros(Float64, length(x), length(x))
        _hess!(H, x)
        return H
    end
    _FIM! = (FIM, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        src.FIM!(_FIM_full, xest_full)
        _map_matrix!(FIM, _FIM_full, imap)
        return nothing
    end
    _FIM = (x) -> begin
        FIM = zeros(Float64, length(x), length(x))
        _FIM!(FIM, x)
        return FIM
    end
    return PEtabODEProblem(
        _nllh, _chi2, _grad!, _grad, _hess!, _hess, _FIM!, _FIM, _nllh_grad, _prior,
        _grad_prior, _hess_prior, _simulated_values, _residuals, prob.probinfo,
        prob.model_info, nestimate_new, xnames_new, xnominal_new, xnominal_transformed_new,
        lb_new, ub_new
    )
end

function _remake_experiments(prob::PEtabODEProblem, experiments::Vector{Symbol})
    conditions_v1 = Any[]
    for experiment in experiments
        _check_experiment_id(nothing, experiment, prob.model_info)
        simulation_id = _get_simulation_id(nothing, experiment, prob.model_info)
        pre_equilibration_id = _get_pre_equilibration_id(nothing, experiment, prob.model_info)
        if isnothing(pre_equilibration_id)
            push!(conditions_v1, simulation_id)
        else
            push!(conditions_v1, pre_equilibration_id => simulation_id)
        end
    end

    # Ensure correct types for _remake_conditions
    conditions_v1 = [conditions_v1...]
    return _remake_conditions(prob, conditions_v1)
end

function _remake_conditions(prob::PEtabODEProblem, conditions::Vector{Symbol})
    @unpack simulation_info = prob.model_info
    if simulation_info.has_pre_equilibration
        throw(PEtabFormatError("This PEtab problem uses pre-equilibration, so \
            `conditions` passed to `remake`  must specify pre-eq/simulation pairs, e.g. \
             `[:pre_id1 => :sim_id1), ...]`."))
    end

    for simulation_id in conditions
        _check_condition_ids(simulation_id, nothing, prob.model_info)
    end

    valid_ids = simulation_info.conditionids[:experiment]
    index_delete = findall(x -> x ∉ conditions, valid_ids)
    return _remake_condition_ids(prob, index_delete)
end
function _remake_conditions(
        prob::PEtabODEProblem, conditions::Vector{<:Pair}
    )::PEtabODEProblem
    @unpack simulation_info = prob.model_info

    if !simulation_info.has_pre_equilibration
        throw(PEtabFormatError("This PEtab problem does not use pre-equilibration, so \
            `conditions` passed to `remake`  must be a `firstVector{Symbol}`, e.g. \
            `[:cond1, :cond2, ...]`."))
    end

    for experiment_id in conditions
        _check_condition_ids(experiment_id.second, experiment_id.first, prob.model_info)
    end

    valid_ids = simulation_info.conditionids[:experiment]
    experiment_ids = [_get_experiment_id(e.second, e.first) for e in conditions]
    index_delete = findall(x -> x ∉ experiment_ids, valid_ids)
    return _remake_condition_ids(prob, index_delete)
end

function _remake_condition_ids(prob::PEtabODEProblem, index_delete::Vector{Int64})
    @unpack model_info, probinfo = prob
    @unpack simulation_info = model_info

    # Only the set of simulated conditions differs between prob and the remade problem.
    # Rather than a deepcopy of prob (which copies the ODEProblem, model, callbacks and
    # every cache), only the state that must differ is rebuilt, and the remaining
    # read-only template data is shared. Besides being much faster, this also avoids
    # deepcopy-ing the RuntimeGeneratedFunctions held by the callbacks and the ODEProblem;
    # a nested deepcopy clones their cached body expression, which breaks the function
    # once the original prob is garbage collected (see _parse_events)
    conditionids = Dict{Symbol, Vector{Symbol}}()
    for (key, ids) in simulation_info.conditionids
        conditionids[key] = copy(ids)
    end
    for key in (:simulation, :pre_equilibration, :experiment)
        deleteat!(conditionids[key], index_delete)
    end

    # ODESolutions and could_solve are per-evaluation output. They start out empty, as for
    # a newly created problem, so that the two problems do not share solution storage
    _simulation_info = @set simulation_info.conditionids = conditionids
    _simulation_info = @set _simulation_info.odesols = Dict{Symbol, ODESolution}()
    _simulation_info = @set _simulation_info.odesols_derivatives = Dict{
        Symbol, ODESolution,
    }()
    _simulation_info = @set _simulation_info.odesols_preeq = Dict{
        Symbol, Union{ODESolution, SciMLBase.NonlinearSolution},
    }()
    _simulation_info = @set _simulation_info.could_solve = [true]

    # petab_measurements holds the simulated values, chi2 and residuals written by each
    # evaluation, and priors tracks which parameters to skip. Both are small, and are
    # copied to keep the problems independent
    _model_info = @set model_info.simulation_info = _simulation_info
    _model_info = @set _model_info.petab_measurements = deepcopy(
        model_info.petab_measurements
    )
    _model_info = @set _model_info.priors = deepcopy(model_info.priors)

    # The cache is scratch space written during each evaluation, so the remade problem
    # needs its own. Building a new cache is cheaper than deepcopy-ing the existing one
    _cache = PEtabODEProblemCache(
        probinfo.gradient_method, probinfo.hessian_method, probinfo.FIM_method,
        probinfo.sensealg, _model_info, model_info.model.ml_models,
        probinfo.split_over_conditions, probinfo.odeproblem
    )
    _probinfo = @set probinfo.cache = _cache

    return _petab_odeproblem(_probinfo, _model_info)
end

function _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
    xest_full = convert.(eltype(x), _xest_full)
    xest_full[ix_fixed] .= x_fixed
    xest_full[imap] .= x
    return xest_full
end

function _to_component_array(x)::ComponentVector{Float64}
    if isempty(x)
        return ComponentVector{Float64}()
    else
        return x
    end
end

function _map_matrix!(x_subset, x_full, imap)
    for (i1, i2) in pairs(imap)
        for (j1, j2) in pairs(imap)
            x_subset[i1, j1] = x_full[i2, j2]
        end
    end
    return
end
