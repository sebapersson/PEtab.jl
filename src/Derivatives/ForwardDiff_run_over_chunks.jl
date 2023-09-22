function forwarddiff_gradient_chunks(f::F, Δx, x, ::ForwardDiff.Chunk{C}; n_forward_passes=nothing) where {F, C}

    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{C}(), nothing)
    N = length(x)
    xdual = cfg.duals
    seeds = cfg.seeds
    ForwardDiff.seed!(xdual, x)

    _n_forward_passes = isnothing(n_forward_passes) ? Int64(ceil(length(x) / C)) : n_forward_passes
    for i in 1:_n_forward_passes
        i_start = (i-1)*C + 1
        i_end = i_start + C
        if i_end ≤ (N+1)
            ForwardDiff.seed!(xdual, x, i_start, seeds)
            ydual = f(xdual)
            ForwardDiff.extract_gradient_chunk!(Nothing, Δx, ydual, i_start, C)
            ForwardDiff.seed!(xdual, x, i_start)
        else
            lastchunksize = N - C*(i-1)
            ForwardDiff.seed!(xdual, x, i_start, seeds, lastchunksize)
            _ydual = f(xdual)
            ForwardDiff.extract_gradient_chunk!(Nothing, Δx, _ydual, i_start, lastchunksize)
        end
    end
end


# Only works with inplace function f!
function forwarddiff_jacobian_chunks(f!::F, y, J, x, ::ForwardDiff.Chunk{C}; n_forward_passes=nothing) where {F, C}

    J .= 0.0
    cfg = ForwardDiff.JacobianConfig(f!, y, x, ForwardDiff.Chunk{C}(), nothing)

    # figure out loop bounds
    N = length(x)
    ydual, xdual = cfg.duals
    ForwardDiff.seed!(xdual, x)
    seeds = cfg.seeds
    Δx_reshaped = ForwardDiff.reshape_jacobian(J, ydual, xdual)

    _n_forward_passes = isnothing(n_forward_passes) ? Int64(ceil(length(x) / C)) : n_forward_passes
    for i in 1:_n_forward_passes
        i_start = (i-1)*C + 1
        i_end = i_start + C
        if i_end ≤ (N+1)
            ForwardDiff.seed!(xdual, x, i_start, seeds)
            f!(ForwardDiff.seed!(ydual, y), xdual)
            ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, i_start, C)
            ForwardDiff.seed!(xdual, x, i_start)
        else
            lastchunksize = N - C*(i-1)
            ForwardDiff.seed!(xdual, x, i_start, seeds, lastchunksize)
            f!(ForwardDiff.seed!(ydual, y), xdual)
            ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, i_start, lastchunksize)
        end
    end
end