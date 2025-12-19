"""
    remake(prob::PEtabODEProblem, condition_ids) -> PEtabODEProblem

Create a new `PEtabODEProblem` that uses a subset of the original problem’s simulation
condition ids, as specified by `condition_ids`.

Intended for performant subsetting (e.g. evaluating `nllh/grad/hessian` on only a
subset of simulation conditions). It is faster than constructing a new `PEtabODEProblem`
since generated code is reused (no recompilation).

# Arguments
- `prob`: The `PEtabODEProblem` to subset.
- `condition_ids`: Simulation condition identifiers to keep (must match the condition ids
  defined in `prob`). If empty (default), `prob` is returned unchanged. The required format
  depends on whether the model has pre-equilibration:
  - No pre-equilibration: `Vector{Symbol}` of simulation condition ids.
  - With pre-equilibration: `Vector{NamedTuple}` of pre-eq/simulation pairs, e.g.
    `[(pre_eq = :pre_id1, simulation = :sim_id1), ...]`.

## Example
```julia
# keep only simulation conditions :cond1 and :cond3
prob_sub = remake(prob; condition_ids = [:cond1, :cond3])
```
"""
function remake(prob::PEtabODEProblem, condition_ids::Vector{Symbol})::PEtabODEProblem
    if isempty(condition_ids)
        return deepcopy(prob)
    end

    @unpack simulation_info = prob.model_info
    if simulation_info.has_pre_equilibration
        throw(PEtabFormatError("This PEtab problem uses pre-equilibration, so \
            `condition_ids` passed to `remake`  must specify pre-eq/simulation pairs, e.g. \
             `[(pre_eq = :pre_id1, simulation = :sim_id1), ...]`."))
    end

    valid = simulation_info.conditionids[:experiment]
    for cid in condition_ids
        cid in valid && continue
        throw(PEtabFormatError("Simulation condition id `$(cid)` in `condition_ids` \
            passed to `remake` is not defined in this PEtab problem. Valid ids are $(valid)"
        ))
    end

    index_delete = findall(x -> x ∉ condition_ids, valid)
    return _remake_condition_ids(prob, index_delete)
end
function remake(prob::PEtabODEProblem, condition_ids::Vector{<:NamedTuple})::PEtabODEProblem
    if isempty(condition_ids)
        return deepcopy(prob)
    end

    @unpack simulation_info = prob.model_info
    if !simulation_info.has_pre_equilibration
        throw(PEtabFormatError("This PEtab problem does not use pre-equilibration, so \
            `condition_ids` passed to `remake`  must be a `Vector{Symbol}`, e.g. \
            `[:cond1, :cond2, ...]`."))
    end

    for cid in condition_ids
        (cid isa @NamedTuple{pre_eq::Symbol, simulation::Symbol} ||
         cid isa @NamedTuple{simulation::Symbol, pre_eq::Symbol}) && continue
        throw(PEtabFormatError("For PEtab problems with pre-equilibration, each entry in \
            `condition_ids` passed to `remake` must be a `NamedTuple` with keys `pre_eq` \
             and `simulation`, e.g. `(pre_eq = :pre_id1, simulation = :sim_id1)`. \
            Got $(cid)."))
    end

    _pre_eq_ids = getfield.(condition_ids, :pre_eq)
    _simulation_ids = getfield.(condition_ids, :simulation)
    _experiment_ids = Symbol.(string.(_pre_eq_ids) .* string.(_simulation_ids))

    valid = simulation_info.conditionids[:experiment]
    for (i, experiment_id) in pairs(_experiment_ids)
        experiment_id in valid && continue
        throw(PEtabFormatError("Condition pair $(condition_ids[i]) in `condition_ids` \
            passed to `remake` is not defined in the original PEtab problem."))
    end

    index_delete = findall(x -> x ∉ _experiment_ids, valid)
    return _remake_condition_ids(prob, index_delete)
end
function remake(prob::PEtabODEProblem, xchange::Dict)::PEtabODEProblem
    # It only makes sense to remake (from compilation point if view) if parameters that
    # before were to be estimated are set to fixated. In PEtab-select setting some
    # parameters can also be set to estimate, these can be removed here.
    for xid in keys(xchange)
        if xchange[xid] == "estimate" && !(xid in prob.xnames)
            throw(PEtabInputError("When remaking a PEtabODEProblem new parameters, here " *
                                  "$xid, which is not set to be estimated in the PEtab " *
                                  "table cannot be set to be estimated. Instead build " *
                                  "a new PEtabODEProblem with $xid set to be estimated"))
        elseif xchange[xid] == "estimate"
            delete!(xchange, xid)
        end
    end

    # Map the fixated values to the parameter scale (they are assumed to be on the linear)
    # scale (e.g. not log10 which might be needed by the PEtab-problem)
    @unpack model_info, probinfo = prob
    split_over_conditions = probinfo.split_over_conditions
    xids_fixate = keys(xchange) |> collect
    x_fixate = zeros(Float64, length(xids_fixate))
    for (i, xid) in pairs(xids_fixate)
        scale = model_info.xindices.xscale[xid]
        x_fixate[i] = transform_x(xchange[xid], scale; to_xscale = true)
    end

    # In case we fixate more parameters than there are chunks we might only want to
    # evaluate ForwardDiff over a subset of chunks. To this end we here make sure
    # "fixed" parameter are moved to the end of the parameter vector to not run ForwardDiff
    # over these
    xids_dynamic = model_info.xindices.xids[:dynamic]
    ixdynamic_fixate = Int64[]
    for xid in xids_fixate
        ix = findfirst(x -> x == xid, xids_dynamic)
        isnothing(ix) && continue
        push!(ixdynamic_fixate, ix)
    end
    if !isempty(ixdynamic_fixate)
        k = 1
        # Make sure the parameter which are to be "estimated" end up in the front of the
        # parameter vector when running ForwardDiff
        for i in eachindex(xids_dynamic)
            i in ixdynamic_fixate && continue
            probinfo.cache.xdynamic_input_order[k] = i
            probinfo.cache.xdynamic_output_order[i] = k
            k += 1
        end
        # Make sure the parameter which are fixated ends up in the end of the parameter
        # vector for ForwardDiff
        for i in eachindex(xids_dynamic)
            !(i in ixdynamic_fixate) && continue
            probinfo.cache.xdynamic_input_order[k] = i
            probinfo.cache.xdynamic_output_order[i] = k
            k += 1
        end
        probinfo.cache.nxdynamic[1] = length(xids_dynamic) - length(ixdynamic_fixate)

        # Aviod  problems with autodiff=true for ODE solvers when computing gradient
        solver_gradient = probinfo.solver_gradient.solver
        if solver_gradient isa Rodas5P
            probinfo.solver_gradient.solver = Rodas5P(autodiff = false)
        elseif solver_gradient isa Rodas5
            probinfo.solver_gradient.solver = Rodas5(autodiff = false)
        elseif solver_gradient isa Rodas4
            probinfo.solver_gradient.solver = Rodas4(autodiff = false)
        elseif solver_gradient isa Rodas4P
            probinfo.solver_gradient.solver = Rodas4P(autodiff = false)
        end
    else
        probinfo.cache.xdynamic_input_order .= 1:length(xids_dynamic)
        probinfo.cache.xdynamic_output_order .= 1:length(xids_dynamic)
        probinfo.cache.nxdynamic[1] = length(xids_dynamic)
    end

    # Parameters and other things for the remade problem. ix_names are needed here as
    # in the PEtabODEProblem nominal values and bounds are ComponentArrays.
    # empty_to_component_array is needed as a vector is returned
    ix = findall(x -> !(x in xids_fixate), prob.xnames)
    ix_names = propertynames(prob.xnominal)[ix]
    ix_names_ps = propertynames(prob.xnominal_transformed)[ix]
    lb = prob.lower_bounds[ix_names_ps] |> _to_component_array
    ub = prob.upper_bounds[ix_names_ps] |> _to_component_array
    xnames = prob.xnames[ix]
    xnominal = prob.xnominal[ix_names] |> _to_component_array
    xnominal_transformed = prob.xnominal_transformed[ix_names_ps] |> _to_component_array
    nestimate = length(xnames)

    # Set priors to be skipped (first reset to not skip any evaluated parameters)
    priors = model_info.priors
    filter!(x -> isnothing(x), priors.skip)
    for xid in xids_fixate
        push!(priors.skip, xid)
    end

    # Needed for the new problem (as under the hood we still use the full Hessian and
    # gradient, so these need to be pre-allocated)
    _xest_full = similar(prob.xnominal) |> collect
    _grad_full = similar(prob.xnominal) |> collect
    _hess_full = zeros(Float64, length(_xest_full), length(_xest_full))
    _FIM_full = zeros(Float64, length(_xest_full), length(_xest_full))
    ix_fixate = [findfirst(x -> x == id, prob.xnames) for id in xids_fixate]
    imap = [findfirst(x -> x == xnames[i], prob.xnames) for i in eachindex(xnames)]

    # PEtabODEProblem functions
    _prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        return prob.prior(xest_full)
    end
    _grad_prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        g = prob.grad_prior(xest_full)
        return g[imap]
    end
    _hess_prior = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        _H = prob.grad_hess(xest_full)
        H = zeros(eltype(H), length(x), length(x))
        for (i1, i2) in pairs(imap)
            for (j1, j2) in pairs(imap)
                H[i1, j1] = _H[i2, j2]
            end
        end
        return H
    end
    _nllh = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        return prob.nllh(xest_full)
    end
    _simulated_values = (x; as_array = false) -> begin
        xest_full = _set_xest(xest_full, x, ix_fixate, x_fixate, imap)
        return prob.simulated_values(_xest_full)
    end
    _chi2 = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        return prob.chi2(xest_full)
    end
    _residuals = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        return prob.residuals(xest_full)
    end
    _grad! = (g, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        if (probinfo.gradient_method in [:ForwardDiff, :ForwardEquations])
            prob.grad!(_grad_full, xest_full; isremade = true)
        else
            prob.grad!(_grad_full, xest_full)
        end
        g .= _grad_full[imap]
        return nothing
    end
    _grad = (x) -> begin
        g = similar(x)
        _grad!(g, x)
        return g
    end
    _nllh_grad = (x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        nllh, _grad_full = prob.nllh_grad(xest_full)
        return nllh, _grad_full[imap]
    end
    _hess! = (H, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        if (probinfo.hessian_method == :GaussNewton) && split_over_conditions == false
            prob.hess!(_hess_full, xest_full; isremade = true)
        else
            prob.hess!(_hess_full, xest_full)
        end
        # Can use double index with first and second
        for (i1, i2) in pairs(imap)
            for (j1, j2) in pairs(imap)
                H[i1, j1] = _hess_full[i2, j2]
            end
        end
        return nothing
    end
    _hess = (x) -> begin
        H = zeros(Float64, length(x), length(x))
        _hess!(H, x)
        return H
    end
    _FIM! = (FIM, x) -> begin
        xest_full = _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
        prob.FIM!(_FIM_full, xest_full)
        # Can use double index with first and second
        @inbounds for (i1, i2) in pairs(imap)
            for (j1, j2) in pairs(imap)
                FIM[i1, j1] = _FIM_full[i2, j2]
            end
        end
        return nothing
    end
    _FIM = (x) -> begin
        FIM = zeros(Float64, length(x), length(x))
        _FIM!(FIM, x)
        return FIM
    end
    return PEtabODEProblem(_nllh, _chi2, _grad!, _grad, _hess!, _hess, _FIM!, _FIM,
                           _nllh_grad, _prior, _grad_prior, _hess_prior, _simulated_values,
                           _residuals, prob.probinfo, prob.model_info, nestimate, xnames,
                           xnominal, xnominal_transformed, lb, ub)
end

function _remake_condition_ids(prob::PEtabODEProblem, index_delete::Vector{Int64})
    _prob = deepcopy(prob)
    conditionids = _prob.model_info.simulation_info.conditionids
    deleteat!(conditionids[:simulation], index_delete)
    deleteat!(conditionids[:pre_equilibration], index_delete)
    deleteat!(conditionids[:experiment], index_delete)
    return _prob
end

function _set_xest(_xest_full, x, ix_fixate, x_fixate, imap)
    xest_full = convert.(eltype(x), _xest_full)
    xest_full[ix_fixate] .= x_fixate
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
