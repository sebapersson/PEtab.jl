function PEtab._reshape_array(x, mapping)
    dims_out = size(x)[last.(mapping)]
    xout = reshape(deepcopy(x), dims_out)
    for i in eachindex(Base.CartesianIndices(x))
        inew = zeros(Int64, length(i.I))
        for j in eachindex(i.I)
            inew[j] = i.I[mapping[j].second]
        end
        xout[inew...] = x[i]
    end
    return xout
end

function PEtab._setup_nnmodels(nnmodels::Dict{Symbol, <:NNModel})::Dict
    rng = Random.default_rng()
    nn = Dict()
    for (netid, nnmodel) in nnmodels
        _, _st = Lux.setup(rng, nnmodel[:net])
        nn[netid] = [_st, nnmodel[:net]]
    end
    return nn
end

function PEtab._get_pnns(nnmodels_in_ode::Dict)::Dict{Symbol, NamedTuple}
    rng = Random.default_rng()
    pnns = Dict()
    for (netid, nnmodel) in nnmodels_in_ode
        _pnn, _ = Lux.setup(rng, nnmodel.nn)
        pnns[netid] = _pnn
    end
    return pnns
end

function PEtab._get_n_net_parameters(nnmodel::PEtab.NNModel)
    return Lux.LuxCore.parameterlength(nnmodel.nn)
end

function PEtab._get_nn_initialparameters(nnmodel::PEtab.NNModel)::ComponentArray
    rng = Random.default_rng(1)
    return Lux.initialparameters(rng, nnmodel.nn) |> ComponentArray .|> Float64
end

function PEtab._reshape_io_data(x::Array{T})::Array{T} where T <: AbstractFloat
    order_py = 1:length(size(x))
    order_jl = reverse(order_py)
    imap = zeros(Int64, length(order_jl))
    for i in eachindex(order_jl)
        imap[i] = findfirst(x -> x == order_jl[i], order_py)
    end
    map = collect(1:length(order_py)) .=> imap
    return PEtab._reshape_array(x, map)
end

function PEtab._get_initialisation_priors(parameters_df::DataFrame)::Vector{Function}
    isempty(parameters_df) && return

    if !(:initializationPriorType in propertynames(parameters_df))
        fsample = (rng, x) -> kaiming_uniform(rng, Float64, size(x)...; gain = 1.0)
        return [fsample for _ in 1:nrow(parameters_df)]
    end

    fsamples = Vector{Function}(undef, nrow(parameters_df))
    for (i, prior) in pairs(parameters_df[!, :initializationPriorType])
        if ismissing(prior)
            fsamples[i] = (rng, x) -> kaiming_uniform(rng, Float64, size(x)...; gain = 1.0)
        elseif prior == "kaimingUniform"
            gain =_parse_prior_parameters(parameters_df[i, :initializationPriorParameters])
            fsamples[i] = (rng, x) -> kaiming_uniform(rng, Float64, size(x)...; gain = gain)
        elseif prior == "kaimingNormal"
            gain =_parse_prior_parameters(parameters_df[i, :initializationPriorParameters])
            fsamples[i] = (rng, x) -> kaiming_normal(rng, Float64, size(x)...; gain = gain)
        elseif prior == "glorotUniform"
            gain =_parse_prior_parameters(parameters_df[i, :initializationPriorParameters])
            fsamples[i] = (rng, x) -> glorot_uniform(rng, Float64, size(x)...; gain = gain)
        elseif prior == "glorotNormal"
            gain =_parse_prior_parameters(parameters_df[i, :initializationPriorParameters])
            fsamples[i] = (rng, x) -> glorot_normal(rng, Float64, size(x)...; gain = gain)
        elseif prior == "normal"
            μ, σ =_parse_prior_parameters(parameters_df[i, :initializationPriorParameters])
            dist = Distributions.Normal(μ, σ)
            fsamples[i] = (rng, x) -> rand(rng, dist, size(x)...)
        elseif prior == "uniform"
            a, b =_parse_prior_parameters(parameters_df[i, :initializationPriorParameters])
            dist = Distributions.Uniform(a, b)
            fsamples[i] = (rng, x) -> rand(rng, dist, size(x)...)
        else
            throw(PEtabInputError("$prior is not a valid prior for sampling initial neural \
                network parameters."))
        end
    end
    return fsamples
end

function _parse_prior_parameters(x::Real)::Float64
    return x
end
function _parse_prior_parameters(x::String)
    return parse.(Float64, split(x, ';'))
end
