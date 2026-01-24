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
function remake(prob::PEtabODEProblem; conditions::Union{Vector{<:Pair}, Vector{Symbol}} = Symbol[], experiments = Symbol[], parameters::Vector{<:Pair{Symbol, <:Real}} = Pair{Symbol, Real}[])::PEtabODEProblem
    if isempty(conditions) && isempty(parameters) && isempty(experiments)
        return deepcopy(prob)
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
function _remake_parameters(prob::PEtabODEProblem, parameters::Vector{<:Pair{Symbol, <:Real}})::PEtabODEProblem
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

    # Priors should not be computed for fixed parameters
    for ix in ix_fixed
        !in(model_info.priors.ix_prior, ix) && continue
        jx = findfirst(x -> x == ix, model_info.priors.ix_prior)
        model_info.priors.skip[jx] = true
    end

    # PEtabODEProblem functions
    _prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return prob.prior(xest_full)
    end
    _grad_prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        g = prob.grad_prior(xest_full)
        return g[imap]
    end
    _hess_prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        _H = prob.grad_hess(xest_full)
        H = zeros(eltype(H), length(x), length(x))
        _map_matrix!(H, _H, imap)
        return H
    end
    _nllh = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return prob.nllh(xest_full)
    end
    _simulated_values = (x; as_array = false) -> begin
        xest_full = _set_xest(xest_full, x, ix_fixed, x_fixed, imap)
        return prob.simulated_values(xest_full)
    end
    _chi2 = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return prob.chi2(xest_full)
    end
    _residuals = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        return prob.residuals(xest_full)
    end
    _grad! = (g, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        prob.grad!(_grad_full, xest_full)
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
        nllh, _grad_full = prob.nllh_grad(xest_full)
        return nllh, _grad_full[imap]
    end
    _hess! = (H, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
        prob.hess!(_H_full, xest_full)
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
        prob.FIM!(_FIM_full, xest_full)
        _map_matrix!(FIM, _FIM_full, imap)
        return nothing
    end
    _FIM = (x) -> begin
        FIM = zeros(Float64, length(x), length(x))
        _FIM!(FIM, x)
        return FIM
    end
    return PEtabODEProblem(_nllh, _chi2, _grad!, _grad, _hess!, _hess, _FIM!, _FIM,
                           _nllh_grad, _prior, _grad_prior, _hess_prior, _simulated_values,
                           _residuals, prob.probinfo, prob.model_info, nestimate_new,
                           xnames_new, xnominal_new, xnominal_transformed_new, lb_new,
                           ub_new)
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
function _remake_conditions(prob::PEtabODEProblem, conditions::Vector{<:Pair})::PEtabODEProblem
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
    _prob = deepcopy(prob)
    conditionids = _prob.model_info.simulation_info.conditionids
    deleteat!(conditionids[:simulation], index_delete)
    deleteat!(conditionids[:pre_equilibration], index_delete)
    deleteat!(conditionids[:experiment], index_delete)
    return _prob
end

function _set_xest(_xest_full, x, ix_fixed, x_fixed, imap)
    xest_full = convert.(eltype(x), _xest_full)
    xest_full[ix_fixed] .= x_fixed
    xest_full[imap] .= x
    return xest_full
end

function _to_component_array(x)::ComponentArray{Float64}
    if isempty(x)
        return ComponentArray{Float64}()
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
end
