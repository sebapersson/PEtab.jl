function forwardDiffGradientChunks(f::F, Δx, x, ::ForwardDiff.Chunk{C}; nForwardPasses=nothing) where {F, C}

    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{C}(), nothing)
    N = length(x)
    xdual = cfg.duals
    seeds = cfg.seeds
    ForwardDiff.seed!(xdual, x)

    _nForwardPasses = isnothing(nForwardPasses) ? Int64(ceil(length(x) / C)) : nForwardPasses
    for i in 1:_nForwardPasses
        iStart = (i-1)*C + 1
        iEnd = iStart + C
        if iEnd ≤ (N+1)
            ForwardDiff.seed!(xdual, x, iStart, seeds)
            ydual = f(xdual)
            ForwardDiff.extract_gradient_chunk!(Nothing, Δx, ydual, iStart, C)
            ForwardDiff.seed!(xdual, x, iStart)
        else
            lastchunksize = N - C*(i-1)
            ForwardDiff.seed!(xdual, x, iStart, seeds, lastchunksize)
            _ydual = f(xdual)
            ForwardDiff.extract_gradient_chunk!(Nothing, Δx, _ydual, iStart, lastchunksize)
        end
    end
end


# Only works with inplace function f!
function forwardDiffJacobianChunks(f!::F, y, J, x, ::ForwardDiff.Chunk{C}; nForwardPasses=nothing) where {F, C}

    J .= 0.0
    cfg = ForwardDiff.JacobianConfig(f!, y, x, ForwardDiff.Chunk{C}(), nothing)

    # figure out loop bounds
    N = length(x)
    ydual, xdual = cfg.duals
    ForwardDiff.seed!(xdual, x)
    seeds = cfg.seeds
    Δx_reshaped = ForwardDiff.reshape_jacobian(J, ydual, xdual)

    _nForwardPasses = isnothing(nForwardPasses) ? Int64(ceil(length(x) / C)) : nForwardPasses
    for i in 1:_nForwardPasses
        iStart = (i-1)*C + 1
        iEnd = iStart + C
        if iEnd ≤ (N+1)
            ForwardDiff.seed!(xdual, x, iStart, seeds)
            f!(ForwardDiff.seed!(ydual, y), xdual)
            ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, iStart, C)
            ForwardDiff.seed!(xdual, x, iStart)
        else
            lastchunksize = N - C*(i-1)
            ForwardDiff.seed!(xdual, x, iStart, seeds, lastchunksize)
            f!(ForwardDiff.seed!(ydual, y), xdual)
            ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, iStart, lastchunksize)
        end
    end
end