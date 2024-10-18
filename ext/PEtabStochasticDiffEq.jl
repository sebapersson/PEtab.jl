module PEtabStochasticDiffEq

using PEtab
using StochasticDiffEq
using ModelingToolkit
using ComponentArrays

function PEtab.llh(u::AbstractVector, p, it::Int64,
                   measurements_info::PEtab.MeasurementsInfo)::Float64
    @unpack xobservables, xnoise, xnondynamic, nominal_values, obsids, h, sd,
    mapxnoise, mapxobservable, measurements, imeasurements_t,
    measurement_transforms = measurements_info
    t, nllh = measurements_info.t[it], 0.0
    for (j, imeasurement) in pairs(imeasurements_t[it])
        y = measurements[it][j]
        obsid = obsids[imeasurement]

        h = PEtab._h(u, t, p, xobservables, xnondynamic, measurements_info.h,
                     mapxnoise[imeasurement], obsid, nominal_values)
        h_transformed = PEtab.transform_observable(h, measurement_transforms[imeasurement])
        σ = PEtab._sd(u, t, p, xnoise, xnondynamic, measurements_info.sd,
                      mapxobservable[imeasurement], obsid, nominal_values)

        residual = (h_transformed - y) / σ
        nllh += PEtab._nllh_obs(residual, σ, y, measurement_transforms[imeasurement])
    end
    return nllh * -1
end

function PEtab.SDESolver(alg; dt = nothing, adapt::Bool = false)
    if adapt == true && StochasticDiffEq.isadaptive(alg) == false
        throw(PEtab.PEtabInputError("For $alg adapt was set to true, but, $alg is \
                                     not an Adaptive Solver"))
    end
    if isnothing(dt) && adapt == false
        throw(PEtab.PEtabInputError("For alg = $alg either adapt must be set to true, \
                                     or a step-length dt must be provided"))
    end
    return PEtab.SDESolver(alg, dt, adapt)
end

function PEtab.PEtabSDEProblem(model::PEtab.PEtabModel, sde_solver::PEtab.SDESolver;
                               verbose::Bool = false)
    PEtab._logging(:Build_PEtabSDEProblem, verbose; name = model.name)

    # Information needed to compute the likelihood for each simulation condition
    model_info = PEtab.ModelInfo(model, nothing, nothing)

    # Measurement data stored in a format accesiable for partice filters
    # TODO: Make a Dict for simulation conditions
    cid = model_info.simulation_info.conditionids[:experiment][1]
    measurements_info = PEtab.MeasurementsInfo(model_info, cid)

    PEtab._logging(:Build_SDEProblem, verbose)
    btime = @elapsed begin
        sprob = _get_sdeproblem(model)
    end
    PEtab._logging(:Build_SDEProblem, verbose; time = btime)

    # Relevant information for the unknown model parameters
    xnames = model_info.xindices.xids[:estimate]
    xnames_ps = model_info.xindices.xids[:estimate_ps]
    _xnominal = PEtab._get_xnominal(model_info, xnames, false)
    _xnominal_transformed = PEtab._get_xnominal(model_info, xnames, true)
    xnominal = ComponentArray(; (xnames .=> _xnominal)...)
    xnominal_transformed = ComponentArray(; (xnames_ps .=> _xnominal_transformed)...)
    return PEtab.PEtabSDEProblem(model_info, measurements_info, sde_solver, sprob, xnames,
                                 xnames_ps, xnominal, xnominal_transformed)
end

function PEtab._set_x_measurements_info!(prob::PEtab.PEtabSDEProblem, x)::Nothing
    xindices = prob.model_info.xindices
    measurements_info = prob.measurements_info
    _, xobservable, xnoise, xnondynamic = PEtab.split_x(x, xindices)

    xnoise_ps = PEtab.transform_x(xnoise[:], xindices.xids[:noise], xindices)
    xobservable_ps = PEtab.transform_x(xobservable[:], xindices.xids[:observable], xindices)
    xnondynamic_ps = PEtab.transform_x(xnondynamic[:], xindices.xids[:nondynamic], xindices)

    measurements_info.xnoise .= xnoise_ps
    measurements_info.xobservables .= xobservable_ps
    measurements_info.xnondynamic .= xnondynamic_ps
    return nothing
end

function _get_sdeproblem(model::PEtabModel)::SDEProblem
    @unpack sys_mutated, speciemap, parametermap = model
    u0map_tmp = zeros(Float64, length(model.speciemap))
    _sprob = SDEProblem(sys_mutated, u0map_tmp, [0.0, 5e3], parametermap)
    if _sprob.p isa ModelingToolkit.MTKParameters
        _p = _sprob.p.tunable .|> Float64
        sprob = remake(_sprob, p = _p, u0 = Float64.(_sprob.u0))
    else
        sprob = remake(_sprob, p = Float64.(_sprob.p), u0 = Float64.(_sprob.u0))
    end
    return sprob
end

end
