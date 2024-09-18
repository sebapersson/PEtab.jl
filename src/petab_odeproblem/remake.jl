"""
    remake(prob::PEtabODEProblem, xchange::Dict)::PEtabODEProblem

Fixate model parameters for a given PEtabODEProblem without recompiling the problem.

This function allows you to modify parameters without the need to recompile the underlying code, resulting in reduced
latency. To fixate the parameter k1, you can use `xchange=Dict(:k1 => 1.0)`.

If model derivatives are computed using ForwardDiff.jl with a chunk-size of N, the new PEtabODEProblem will only
evaluate the necessary number of chunks of size N to compute the full gradient for the remade problem.
"""
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
