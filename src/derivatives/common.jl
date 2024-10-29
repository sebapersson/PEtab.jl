function _G(u::AbstractVector, p, t::T, i::Integer, imeasurements_t_cid::Vector{Vector{Int64}}, model_info::ModelInfo, xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic_mech::Vector{T}, xnn::Dict{Symbol, ComponentArray}, residuals::Bool) where {T <: AbstractFloat}
    @unpack petab_measurements, xindices, petab_parameters, model = model_info
    @unpack measurement_transforms, observable_id = petab_measurements
    nominal_values = petab_parameters.nominal_value
    out = 0.0
    for imeasurement in imeasurements_t_cid[i]
        obsid = observable_id[imeasurement]
        mapxnoise = xindices.mapxnoise[imeasurement]
        mapxobservable = xindices.mapxobservable[imeasurement]
        h = _h(u, t, p, xobservable, xnondynamic_mech, xnn, model.h, mapxobservable, obsid,
               nominal_values, model.nn)
        h_transformed = transform_observable(h, measurement_transforms[imeasurement])
        σ = _sd(u, t, p, xnoise, xnondynamic_mech, xnn, model.sd, mapxnoise, obsid,
                nominal_values, model.nn)

        y_transformed = petab_measurements.measurement_transformed[imeasurement]
        residual = (h_transformed - y_transformed) / σ
        if residuals == true
            out += residual
            continue
        end
        out += _nllh_obs(residual, σ, y_transformed, measurement_transforms[imeasurement])
    end
    return out
end

function _get_∂G∂_!(model_info::ModelInfo, cid::Symbol, xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic_mech::Vector{T}, xnn::Dict{Symbol, ComponentArray}; residuals::Bool = false)::NTuple{2, Function} where {T <: AbstractFloat}
    if residuals == false
        it = model_info.simulation_info.imeasurements_t[cid]
        ∂G∂u! = (out, u, p, t, i) -> begin
            _fu = (u) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, false)
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i) -> begin
            _fp = (p) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, false)
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    else
        ∂G∂u! = (out, u, p, t, i, it) -> begin
            _fu = (u) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, true)
            end
            ForwardDiff.gradient!(out, _fu, u)
            return nothing
        end
        ∂G∂p! = (out, u, p, t, i, it) -> begin
            _fp = (p) -> begin
                _G(u, p, t, i, it, model_info, xnoise, xobservable, xnondynamic_mech, xnn, true)
            end
            ForwardDiff.gradient!(out, _fp, p)
            return nothing
        end
    end
    return ∂G∂u!, ∂G∂p!
end

# Adjust the gradient from linear scale to current scale for x-vector
function grad_to_xscale!(grad_xscale, grad_linscale::Vector{T}, ∂G∂p::Vector{T},
                         xdynamic::Vector{T}, xindices::ParameterIndices, simid::Symbol;
                         sensitivites_AD::Bool = false, nn_pre_ode::Bool = false,
                         adjoint::Bool = false)::Nothing where {T <: AbstractFloat}
    @unpack dynamic_to_sys, sys_to_dynamic, sys_to_dynamic_nn = xindices.map_odeproblem
    @unpack xids, xscale = xindices
    @unpack ix_sys, ix_dynamic = xindices.maps_conidition_id[simid]
    # Neural net parameters for a neural net before the ODE dynamics. These gradient
    # components only considered for ForwardEquations full AD, where it is not possible to
    # use the chain-rule to separately differentitate each componenent, and these parameters
    # are differentiated using full AD.
    if nn_pre_ode == true
        xi = xindices.xindices_dynamic[:nn_pre_ode]
        @views grad_xscale[xi] .= grad_linscale[xi]
    end
    # Neural net parameters. These should not be transformed (are on linear scale),
    # For everything except sensitivites_AD these are provided in the order of odeproblem.p
    # and are mapped via sys_to_dynamic_nn
    xi = xindices.xindices_dynamic[:nn_in_ode]
    if sensitivites_AD == true
        @views grad_xscale[xi] .= grad_linscale[xi]
    else
        @views grad_xscale[xi] .= grad_linscale[sys_to_dynamic_nn]
    end

    # Adjust for parameters that appear in each simulation condition (not unique to simid).
    # Note that ∂G∂p is on the scale of ODEProblem.p which might not be the same scale
    # as parameters appear in the gradient on linear-scale
    if sensitivites_AD == true
        grad_p1 = grad_linscale[dynamic_to_sys] .+ ∂G∂p[sys_to_dynamic]
    else
        grad_p1 = grad_linscale[sys_to_dynamic] .+ ∂G∂p[sys_to_dynamic]
    end
    @views _grad_to_xscale!(grad_xscale[dynamic_to_sys], grad_p1, xdynamic[dynamic_to_sys],
                            xids[:dynamic_mech][dynamic_to_sys], xscale)

    # For forward sensitives via autodiff ∂G∂p is on the same scale as ode_problem.p, while
    # S-matrix is on the same scale as xdynamic. To be able to handle condition specific
    # parameters mapping to several ode_problem.p parameters the sensitivity matrix part
    # and ∂G∂p must be treated seperately. Further, as condition specific variables can map
    # to several parameters in the ODESystem, the for-loop is needed
    if sensitivites_AD == true
        ix = unique(ix_dynamic)
        @views _grad_to_xscale!(grad_xscale[ix], grad_linscale[ix], xdynamic[ix],
                                xids[:dynamic_mech][ix], xscale)
        @views out = _grad_to_xscale(∂G∂p[ix_sys], xdynamic[ix_dynamic],
                                     xids[:dynamic_mech][ix_dynamic], xscale)
        for (i, imap) in pairs(ix)
            grad_xscale[imap] += out[i]
        end
    end

    # Here both ∂G∂p and grad_linscale are on the same scale a ODEProblem.p. Again
    # condition specific variables can map to several parameters in the ODESystem,
    # thus the for-loop
    if adjoint == true || sensitivites_AD == false
        @views _grad_to_xscale!(grad_xscale[ix_sys], grad_linscale[ix_sys],
                                xdynamic[ix_dynamic], xids[:dynamic_mech][ix_dynamic], xscale)
        @views out = _grad_to_xscale(grad_linscale[ix_sys] .+ ∂G∂p[ix_sys],
                                     xdynamic[ix_dynamic],
                                     xids[:dynamic_mech][ix_dynamic], xscale)
        for (i, imap) in pairs(ix_sys)
            grad_xscale[imap] += out[i]
        end
    end
    return nothing
end

function _grad_to_xscale!(grad_xscale::AbstractVector{T}, grad_linscale::AbstractVector{T},
                          x::AbstractVector{T}, xids::AbstractVector{Symbol},
                          xscale::Dict{Symbol, Symbol})::Nothing where {T <: AbstractFloat}
    for (i, xid) in pairs(xids)
        grad_xscale[i] += _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return nothing
end

function _grad_to_xscale(grad_linscale::AbstractVector{T}, x::AbstractVector{T},
                         xids::AbstractVector{Symbol},
                         xscale::Dict{Symbol, Symbol})::Vector{T} where {T <: AbstractFloat}
    grad_xscale = similar(grad_linscale)
    for (i, xid) in pairs(xids)
        grad_xscale[i] = _grad_to_xscale(grad_linscale[i], x[i], xscale[xid])
    end
    return grad_xscale
end
function _grad_to_xscale(grad_linscale_val::T, x_val::T,
                         xscale::Symbol)::T where {T <: AbstractFloat}
    if xscale === :log10
        return log(10) * grad_linscale_val * x_val
    elseif xscale === :log
        return grad_linscale_val * x_val
    elseif xscale === :lin
        return grad_linscale_val
    end
end

function _could_solveode_nllh(simulation_info::SimulationInfo)::Bool
    for cid in simulation_info.conditionids[:experiment]
        sol = simulation_info.odesols[cid]
        if !(sol.retcode in [ReturnCode.Success, ReturnCode.Terminated])
            return false
        end
    end
    return true
end

function _get_xinput(simid::Symbol, x::Vector{<:AbstractFloat}, ixdynamic_simid, model_info::ModelInfo, probinfo::PEtabODEProblemInfo)
    @unpack xindices, simulation_info = model_info
    ninode, npreode = length(ixdynamic_simid), length(xindices.xids[:nn_pre_ode_outputs])
    xinput = zeros(Float64, ninode + npreode)
    @views xinput[1:ninode] .= x[ixdynamic_simid]
    if isempty(probinfo.nn_pre_ode)
        return xinput
    end
    for (netid, nn_pre_ode) in probinfo.nn_pre_ode[simid]
        map_nn = model_info.xindices.maps_nn_pre_ode[simid][netid]
        outputs = get_tmp(nn_pre_ode.outputs, xinput)
        ix = map_nn.xindices_nn_outputs_grad
        ix .= map_nn.xindices_nn_outputs .+ ninode
        @views xinput[ix] .= outputs
    end
    return xinput
end

function _split_xinput!(probinfo::PEtabODEProblemInfo, simid::Symbol, model_info::ModelInfo, xinput::AbstractVector, ixdynamic_simid::Vector{Integer})::Nothing
    xdynamic_tot = get_tmp(probinfo.cache.xdynamic_tot, xinput)
    @views xdynamic_tot[ixdynamic_simid] .= xinput[1:length(ixdynamic_simid)]
    if isempty(probinfo.nn_pre_ode)
        return nothing
    end
    for (netid, nn_pre_ode) in probinfo.nn_pre_ode[simid]
        map_nn =  model_info.xindices.maps_nn_pre_ode[simid][netid]
        outputs = get_tmp(nn_pre_ode.outputs, xinput)
        @views outputs .= xinput[map_nn.xindices_nn_outputs_grad]
    end
    return nothing
end

function _grad_nn_pre_ode!(xdynamic_grad::AbstractVector, simid::Symbol, probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    isempty(probinfo.nn_pre_ode) && return nothing
    for (netid, nn_pre_ode) in probinfo.nn_pre_ode[simid]
        map_nn = model_info.xindices.maps_nn_pre_ode[simid][netid]
        grad_nn_output = probinfo.cache.grad_nn_pre_ode_outputs[map_nn.xindices_nn_outputs]
        ix = model_info.xindices.xindices_dynamic[netid]
        xdynamic_grad[ix] .+= vec(grad_nn_output' * nn_pre_ode.jac_nn)
    end
    return nothing
end
