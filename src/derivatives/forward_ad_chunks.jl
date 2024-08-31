function forwarddiff_gradient_chunks(f::F, Δx, x, ::ForwardDiff.Chunk{C};
                                     nforward_passes = nothing) where {F, C}
    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{C}(), nothing)
    N = length(x)
    xdual = cfg.duals
    seeds = cfg.seeds
    ForwardDiff.seed!(xdual, x)
    nforward = isnothing(nforward_passes) ? Int64(ceil(length(x) / C)) : nforward_passes
    for i in 1:nforward
        istart = (i - 1) * C + 1
        iend = istart + C
        if iend ≤ (N + 1)
            ForwardDiff.seed!(xdual, x, istart, seeds)
            ydual = f(xdual)
            ForwardDiff.extract_gradient_chunk!(Nothing, Δx, ydual, istart, C)
            ForwardDiff.seed!(xdual, x, istart)
        else
            lastchunksize = N - C * (i - 1)
            ForwardDiff.seed!(xdual, x, istart, seeds, lastchunksize)
            _ydual = f(xdual)
            ForwardDiff.extract_gradient_chunk!(Nothing, Δx, _ydual, istart, lastchunksize)
        end
    end
end

function forwarddiff_jacobian_chunks(f!::F, y, J, x, ::ForwardDiff.Chunk{C};
                                     nforward_passes = nothing) where {F, C}
    fill!(J, 0.0)
    cfg = ForwardDiff.JacobianConfig(f!, y, x, ForwardDiff.Chunk{C}(), nothing)

    N = length(x)
    ydual, xdual = cfg.duals
    ForwardDiff.seed!(xdual, x)
    seeds = cfg.seeds
    Δx_reshaped = ForwardDiff.reshape_jacobian(J, ydual, xdual)
    nforward = isnothing(nforward_passes) ? Int64(ceil(length(x) / C)) : nforward_passes
    for i in 1:nforward
        istart = (i - 1) * C + 1
        iend = istart + C
        if iend ≤ (N + 1)
            ForwardDiff.seed!(xdual, x, istart, seeds)
            f!(ForwardDiff.seed!(ydual, y), xdual)
            ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, istart, C)
            ForwardDiff.seed!(xdual, x, istart)
        else
            lastchunksize = N - C * (i - 1)
            ForwardDiff.seed!(xdual, x, istart, seeds, lastchunksize)
            f!(ForwardDiff.seed!(ydual, y), xdual)
            ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, istart,
                                                lastchunksize)
        end
    end
end
