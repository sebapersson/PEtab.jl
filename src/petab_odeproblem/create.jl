function PEtabODEProblem(model::PEtabModel;
                         odesolver::Union{Nothing, ODESolver} = nothing,
                         odesolver_gradient::Union{Nothing, ODESolver} = nothing,
                         ss_solver::Union{Nothing, SteadyStateSolver} = nothing,
                         ss_solver_gradient::Union{Nothing, SteadyStateSolver} = nothing,
                         gradient_method::Union{Nothing, Symbol} = nothing,
                         hessian_method::Union{Nothing, Symbol} = nothing,
                         FIM_method::Union{Nothing, Symbol} = nothing,
                         sparse_jacobian::Union{Nothing, Bool} = nothing,
                         specialize_level = SciMLBase.FullSpecialize,
                         sensealg = nothing,
                         sensealg_ss = nothing,
                         chunksize::Union{Nothing, Int64} = nothing,
                         split_over_conditions::Union{Nothing, Bool} = nothing,
                         reuse_sensitivities::Bool = false,
                         verbose::Bool = false,
                         custom_values::Union{Nothing, Dict} = nothing)::PEtabODEProblem
    _logging(:Build_PEtabODEProblem, verbose; name = model.name)

    # To bookeep parameters, measurements, etc...
    model_info = ModelInfo(model, sensealg, custom_values)
    # All ODE-relevent info for the problem, e.g. solvers, gradient method ...
    probinfo = PEtabODEProblemInfo(model, model_info, odesolver, odesolver_gradient,
                                   ss_solver, ss_solver_gradient, gradient_method,
                                   hessian_method, FIM_method, sensealg, sensealg_ss,
                                   reuse_sensitivities, sparse_jacobian, specialize_level,
                                   chunksize, split_over_conditions, verbose)

    # The prior enters into the nllh, grad, and hessian functions and is evaluated by
    # default (keyword user can toggle). Grad and hess are not inplace, in order to
    # to not overwrite the nllh hess/grad when evaluating total grad/hess
    prior, grad_prior, hess_prior = _get_prior(model_info)

    _logging(:Build_nllh, verbose)
    btime = @elapsed begin
        nllh = _get_nllh(probinfo, model_info, prior, false)
    end
    _logging(:Build_nllh, verbose; time = btime)

    _logging(:Build_gradient, verbose; method = probinfo.gradient_method)
    btime = @elapsed begin
        method = probinfo.gradient_method
        grad!, grad = _get_grad(Val(method), probinfo, model_info, grad_prior)
        nllh_grad = _get_nllh_grad(method, grad, prior, probinfo, model_info)
    end
    _logging(:Build_gradient, verbose; time = btime)

    _logging(:Build_hessian, verbose; method = probinfo.hessian_method)
    btime = @elapsed begin
        hess!, hess = _get_hess(probinfo, model_info, hess_prior)
        FIM!, FIM = _get_hess(probinfo, model_info, hess_prior; FIM = true)
    end
    _logging(:Build_hessian, verbose; time = btime)

    # Useful functions for getting diagnostics
    _logging(:Build_chi2_res_sim, verbose)
    btime = @elapsed begin
        _chi2 = (x; array = false) -> begin
            _ = nllh(x)
            vals = model_info.petab_measurements.chi2_values
            return array == true ? vals : sum(vals)
        end
        _residuals = (x; array = true) -> begin
            _ = nllh(x)
            vals = model_info.petab_measurements.residuals
            return array == true ? vals : sum(vals)
        end
        _simulated_values = (x) -> begin
            _ = nllh(x)
            return model_info.petab_measurements.simulated_values
        end
    end
    _logging(:Build_chi2_res_sim, verbose; time = btime)

    # Relevant information for the unknown model parameters
    xnames = model_info.xindices.xids[:estimate]
    xnames_ps = model_info.xindices.xids[:estimate_ps]
    nestimate = length(xnames)
    lb = _get_bounds(model_info, xnames, :lower)
    ub = _get_bounds(model_info, xnames, :upper)
    xnominal = _get_xnominal(model_info, xnames, xnames_ps, false)
    xnominal_transformed = _get_xnominal(model_info, xnames, xnames_ps, true)

    return PEtabODEProblem(nllh, _chi2, grad!, grad, hess!, hess, FIM!, FIM, nllh_grad,
                           prior, grad_prior, hess_prior, _simulated_values, _residuals,
                           probinfo, model_info, nestimate, xnames, xnominal,
                           xnominal_transformed, lb, ub)
end

function _get_prior(model_info::ModelInfo)::Tuple{Function, Function, Function}
    _prior = let minfo = model_info
        @unpack xindices, priors = minfo
        (x) -> begin
            xnames = xindices.xids[:estimate]
            return prior(x, xnames, priors, xindices)
        end
    end
    _grad_prior = let p = _prior
        x -> ForwardDiff.gradient(p, x)
    end
    _hess_prior = let p = _prior
        x -> ForwardDiff.hessian(p, x)
    end
    return _prior, _grad_prior, _hess_prior
end

function _get_nllh(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   prior::Function, residuals::Bool)::Function
    _nllh = let pinfo = probinfo, minfo = model_info, res = residuals, _prior = prior
        (x; prior = true) -> begin
            _test_ordering(x, minfo.xindices.xids[:estimate_ps])
            _x = x |> collect
            nllh_val = nllh(_x, pinfo, minfo, [:all], false, res)
            if prior == true && res == false
                # nllh -> negative prior
                return nllh_val - _prior(_x)
            else
                return nllh_val
            end
        end
    end
    return _nllh
end

function _get_grad(method, probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   grad_prior::Function)::Tuple{Function, Function}
    if probinfo.gradient_method == :ForwardDiff
        _grad_nllh! = _get_grad_forward_AD(probinfo, model_info)
    end
    if probinfo.gradient_method == :ForwardEquations
        _grad_nllh! = _get_grad_forward_eqs(probinfo, model_info)
    end

    _grad! = let _grad_nllh! = _grad_nllh!, grad_prior = grad_prior
        (g, x; prior = true, isremade = false) -> begin
            _x = x |> collect
            _g = similar(_x)
            _grad_nllh!(_g, _x; isremade = isremade)
            if prior
                # nllh -> negative prior
                _g .+= grad_prior(_x) .* -1
            end
            g .= _g
            return nothing
        end
    end
    _grad = let _grad! = _grad!
        (x; prior = true, isremade = false) -> begin
            gradient = similar(x)
            _grad!(gradient, x; prior = prior, isremade = isremade)
            return gradient
        end
    end
    return _grad!, _grad
end

function _get_hess(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   hess_prior::Function; ret_jacobian::Bool = false,
                   FIM::Bool = false)::Tuple{Function, Function}
    @unpack hessian_method, split_over_conditions, chunksize, cache = probinfo
    @unpack xdynamic_mech = cache
    if FIM == true
        hessian_method = probinfo.FIM_method
    end

    if hessian_method === :ForwardDiff
        _hess_nllh! = _get_hess_forward_AD(probinfo, model_info)
    elseif hessian_method === :BlockForwardDiff
        _hess_nllh! = _get_hess_block_forward_AD(probinfo, model_info)
    elseif hessian_method == :GaussNewton
        _hess_nllh! = _get_hess_gaussnewton(probinfo, model_info, ret_jacobian)
    end

    _hess! = let _hess_nllh! = _hess_nllh!, hess_prior = hess_prior
        (H, x; prior = true, isremade = false) -> begin
            _x = x |> collect
            _H = H |> collect
            if hessian_method == :GassNewton
                _hess_nllh!(_H, _x; isremade = isremade)
            else
                _hess_nllh!(_H, _x)
            end
            if prior && ret_jacobian == false
                # nllh -> negative prior
                _H .+= hess_prior(_x) .* -1
            end
            H .= _H
            return nothing
        end
    end
    _hess = (x; prior = true) -> begin
        if hessian_method == :GaussNewton && ret_jacobian == true
            H = zeros(eltype(x), length(x), length(model_info.petab_measurements.time))
        else
            H = zeros(eltype(x), length(x), length(x))
        end
        _hess!(H, x; prior = prior)
        return H
    end
    return _hess!, _hess
end

function _get_nllh_grad(gradient_method::Symbol, grad::Function, _prior::Function,
                        probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Function
    grad_forward_AD = gradient_method == :ForwardDiff
    grad_forward_eqs = gradient_method == :ForwardEquations
    grad_adjoint = gradient_method == :Adjoint

    _nllh_not_solveode = _get_nllh_not_solveode(probinfo, model_info;
                                                grad_forward_AD = grad_forward_AD,
                                                grad_forward_eqs = grad_forward_eqs,
                                                grad_adjoint = grad_adjoint)
    _nllh_grad = (x; prior = true) -> begin
        _x = x |> collect
        g = grad(x; prior = prior)
        x_notode = @view _x[model_info.xindices.xindices[:not_system_mech]]
        nllh = _nllh_not_solveode(x_notode)
        if prior
            nllh += _prior(_x)
        end
        return nllh, g
    end
    return _nllh_grad
end

function _get_bounds(model_info::ModelInfo, xnames::Vector{Symbol}, which::Symbol)
    @unpack petab_parameters, xindices = model_info
    xnames_not_nn = xnames[findall(x -> !(x in xindices.xids[:nn]), xnames)]
    xnames_nn = xnames[findall(x -> x in xindices.xids[:nn], xnames)]

    # Mechanistic parameters are given as Vector
    ix = [findfirst(x -> x == id, petab_parameters.parameter_id) for id in xnames_not_nn]
    bounds_mechanistic = (xnames_not_nn .=> petab_parameters.lower_bounds[ix]) |> NamedTuple
    # Network parameters are given as ComponentArray
    vals = Vector{Any}(undef, length(xnames_nn))
    for (i, net) in pairs(collect(values(model_info.model.nn)))
        rng = Random.default_rng(1)
        vals[i] = Lux.initialparameters(rng, net[2]) |> ComponentArray
        if which == :lower
            vals[i] .= -10.0
        else
            vals[i] .= 10.0
        end
    end
    bounds_net = (xnames_nn .=> vals) |> NamedTuple
    return merge(bounds_mechanistic, bounds_net) |> ComponentArray
end

function _get_xnominal(model_info::ModelInfo, xnames::Vector{Symbol},
                       xnames_ps::Vector{Symbol}, transform::Bool)
    @unpack petab_parameters, xindices = model_info
    ixnames_not_nn = findall(x -> !(x in xindices.xids[:nn]), xnames)
    xnames_not_nn = xnames[ixnames_not_nn]
    xnames_nn = xnames[findall(x -> x in xindices.xids[:nn], xnames)]

    # Mechanistic parameters are given as Vector
    ix = [findfirst(x -> x == id, petab_parameters.parameter_id) for id in xnames_not_nn]
    xnominal = petab_parameters.nominal_value[ix]
    if transform == true
        transform_x!(xnominal, xnames, xindices, to_xscale = true)
        vals_mechanistic = (xnames_ps[ixnames_not_nn] .=> xnominal) |> NamedTuple
    else
        vals_mechanistic = (xnames[ixnames_not_nn] .=> xnominal) |> NamedTuple
    end
    # Network parameters are given as ComponentArray
    vals = Vector{Any}(undef, length(xnames_nn))
    for (i, net) in pairs(collect(values(model_info.model.nn)))
        rng = Random.default_rng(1)
        vals[i] = Lux.initialparameters(rng, net[2]) |> ComponentArray
        vals[i] .= 0.0
    end
    vals_nn = (xnames_nn .=> vals) |> NamedTuple
    return merge(vals_mechanistic, vals_nn) |> ComponentArray
end

function _test_ordering(x::ComponentArray, xnames_ps::Vector{Symbol})::Nothing
    if !all(propertynames(x) .== xnames_ps)
        throw(PEtabInputError("Input ComponentArray x to the PEtab nllh function \
                               has wrong ordering or parameter names. In x the \
                               parameters must appear in the order of $xnames_ps"))
    end
    return nothing
end
_test_ordering(x::AbstractVector, names_ps::Vector{Symbol})::Nothing = nothing

# TODO: Put in Lux functionality (from here and down)
function _net!(out, x, pnn::DiffCache, inputs::DiffCache, map_nn::NNPreODEMap, nn)::Nothing
    _inputs = get_tmp(inputs, x)
    _pnn = get_tmp(pnn, x)
    _inputs[map_nn.iconstant_inputs] .= map_nn.constant_inputs
    @views _inputs[map_nn.ixdynamic_inputs] .= x[1:map_nn.nxdynamic_inputs]
    @views _pnn .= x[(map_nn.nxdynamic_inputs + 1):end]
    st, net = nn
    out .= net(_inputs, _pnn, st)[1]
    return nothing
end

function _get_nn_pre_ode_x(nnpre::NNPreODE, xdynamic_mech::AbstractVector, pnn::ComponentArray, map_nn::NNPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_nn.nxdynamic_inputs] = xdynamic_mech[map_nn.ixdynamic_mech_inputs]
    @views x[(map_nn.nxdynamic_inputs+1):end] .= pnn
    return x
end

function _get_net_values(mapping_table::DataFrame, netid::Symbol, type::Symbol)::Vector{String}
    dfnet = mapping_table[Symbol.(mapping_table[!, :netId]) .== netid, :]
    if type == :outputs
        dfvals = dfnet[startswith.(string.(dfnet[!, :ioId]), "output"), :]
    elseif type == :inputs
        dfvals = dfnet[startswith.(string.(dfnet[!, :ioId]), "input"), :]
    end
    # Sort to get inputs in order output1, output2, ...
    is = sortperm(string.(dfvals[!, :ioId]), by = x -> parse(Int, match(r"\d+$", x).match))
    return dfvals[is, :ioValue] .|> string
end

function nn_ps_to_tidy(nn, ps::Union{ComponentArray, NamedTuple}, netname::Symbol)::DataFrame
    df_ps = DataFrame()
    for (layername, layer) in pairs(nn.layers)
        ps_layer = ps[layername]
        df_layer = layer_ps_to_tidy(layer, ps_layer, netname, layername)
        df_ps = vcat(df_ps, df_layer)
    end
    return df_ps
end

function set_ps_net!(ps::ComponentArray, df_ps::DataFrame, netname::Symbol, nn)::Nothing
    df_net = df_ps[startswith.( df_ps[!, :parameterId], "$(netname)_"), :]
    for (id, layer) in pairs(nn.layers)
        df_layer = df_net[startswith.(df_net[!, :parameterId], "$(netname)_$(id)_"), :]
        ps_layer = ps[id]
        set_ps_layer!(ps_layer, layer, df_layer)
        ps[id] = ps_layer
    end
    return nothing
end

"""
    layer_ps_to_tidy(layer::Lux.Dense, ps, netname, layername)::DataFrame

Transforms parameters (`ps`) for a Lux layer to a tidy DataFrame `df` with columns `value`
and `parameterId`.

A `Lux.Dense` layer has two sets of parameters, `weight` and optionally `bias`. For
`Dense` and all other form of layers `weight` and `bias` are stored as:

- `netname_layername_weight_ix`: weight for output `i` and input `j`
- `netname_layername_bias_ix`: bias for output `i`

Where `ix` depends on the `Tensor` the parameters are stored in. For example, if
`size(weight) = (5, 2)` `ix` is on the form `ix = i_j`. Here, it is important to note that
the PEtab standard uses Julia tensor. For example, `x = ones(5, 3, 2)` can be thought of
as a Tensor with height 5, width 3 and depth 2. In PyTorch `x` would correspond to
`x = torch.ones(2, 5, 3)`.

For `Dense` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features)`
- `bias` of dimension `(out_features)`
"""
function layer_ps_to_tidy(layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack in_dims, out_dims, use_bias = layer
    df_weight = _ps_weight_to_tidy(ps, (out_dims, in_dims), netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_dims, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `ConvTranspose` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(kernel_size, in_channels, out_channels)`
- `bias` of dimension `(out_channels)`

Note that `kernel_size` for `ConvTranspose` and `Conv` layers can be a `Tuple`, for example
for `Torch` `conv2d` it is a two-dimensional tuple.
"""
function layer_ps_to_tidy(layer::Lux.ConvTranspose, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    df_weight = _ps_weight_to_tidy(ps, (kernel_size..., out_chs, in_chs), netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `Conv` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(kernel_size, out_channels, in_channels)`
- `bias` of dimension `(out_channels)`
"""
function layer_ps_to_tidy(layer::Lux.Conv, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    df_weight = _ps_weight_to_tidy(ps, (kernel_size..., in_chs, out_chs), netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Union{Lux.MaxPool}, ...)::DataFrame

Pooling layers do not have parameters.
"""
function layer_ps_to_tidy(layer::Union{Lux.MaxPool},::Union{NamedTuple, ComponentArray}, ::Symbol, ::Symbol)::DataFrame
    return DataFrame()
end


function set_ps_layer!(ps::ComponentArray, layer::Lux.ConvTranspose, df_ps::DataFrame)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    _set_ps_weight!(ps, (kernel_size..., out_chs, in_chs), df_ps)
    _set_ps_bias!(ps, (out_chs, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(ps::ComponentArray, layer::Lux.Conv, df_ps::DataFrame)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    _set_ps_weight!(ps, (kernel_size..., in_chs, out_chs), df_ps)
    _set_ps_bias!(ps, (out_chs, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, df_ps::DataFrame)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    _set_ps_weight!(ps, (out_dims, in_dims), df_ps)
    _set_ps_bias!(ps, (out_dims, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(::ComponentArray, ::Union{Lux.MaxPool}, ::DataFrame)::Nothing
    return nothing
end

function _set_ps_weight!(ps::ComponentArray, weight_dims, df_ps::DataFrame)::Nothing
    @assert size(ps.weight) == weight_dims "layer size does not match ps.weight size"
    df_weights = df_ps[occursin.("weight_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_weights[!, :parameterId])
        ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-length(weight_dims)+1):end])
        ps.weight[ix...] = df_weights[i, :value]
    end
    return nothing
end

function _set_ps_bias!(ps::ComponentArray, bias_dims, df_ps::DataFrame, use_bias)::Nothing
    use_bias == false && return nothing
    @assert size(ps.bias) == bias_dims "layer size does not match ps.bias size"
    df_bias = df_ps[occursin.("bias_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_bias[!, :parameterId])
        ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-length(bias_dims)+1):end])
        ps.bias[ix...] = df_bias[i, :value]
    end
    return nothing
end

function _ps_weight_to_tidy(ps, weight_dims, netname::Symbol, layername::Symbol)::DataFrame
    iweight = getfield.(findall(x -> true, ones(weight_dims)), :I)
    weight_names =  ["weight" * prod("_" .* string.(ix)) for ix in iweight]
    df_weight = DataFrame(parameterId = "$(netname)_$(layername)_" .* weight_names,
                          value = vec(ps.weight))
    return df_weight
end

function _ps_bias_to_tidy(ps, bias_dims, netname::Symbol, layername::Symbol, use_bias)::DataFrame
    use_bias == false && return DataFrame()
    if length(bias_dims) > 1
        ibias = getfield.(findall(x -> true, ones(bias_dims)), :I)
    else
        ibias = 1:bias_dims[1]
    end
    bias_names =  ["bias" * prod("_" .* string.(ix)) for ix in ibias]
    df_bias = DataFrame(parameterId = "$(netname)_$(layername)_" .* bias_names,
                        value = vec(ps.bias))
    return df_bias
end
