struct SDESolver
    alg::Any
    dt::Union{Nothing, Float64}
    adapt::Bool
end

struct MeasurementsInfo
    t::Vector{Float64}
    measurements::Vector{Vector{Float64}}
    imeasurements_t::Vector{Vector{Int64}}
    obsids::Vector{Symbol}
    xobservables::Vector{Float64}
    xnoise::Vector{Float64}
    xnondynamic_mech::Vector{Float64}
    nominal_values::Vector{Float64}
    mapxnoise::Vector{PEtab.ObservableNoiseMap}
    mapxobservable::Vector{PEtab.ObservableNoiseMap}
    measurement_transforms::Vector{Symbol}
    h::Function
    sd::Function
end
function MeasurementsInfo(model_info::PEtab.ModelInfo, cid::Symbol)
    @unpack simulation_info, model, xindices, petab_measurements = model_info
    @unpack ids, mapxnoise, mapxobservable = xindices
    xobservables = zeros(Float64, length(ids[:observable]))
    xnoise = zeros(Float64, length(ids[:noise]))
    xnondynamic_mech = zeros(Float64, length(ids[:nondynamic_mech]))

    @unpack measurement_transforms, observable_id, measurement_transformed = petab_measurements
    nominval_value = model_info.petab_parameters.nominal_value

    @unpack h, sd = model

    imeasurements_t, t = simulation_info.imeasurements_t[cid], simulation_info.tsaves[cid]
    measurements = imeasurements_t .|> Vector{Float64}
    for i in eachindex(imeasurements_t)
        for j in eachindex(imeasurements_t[i])
            measurements[i][j] = measurement_transformed[imeasurements_t[i][j]]
        end
    end
    return MeasurementsInfo(
        t, measurements, imeasurements_t, observable_id, xobservables,
        xnoise, xnondynamic_mech, nominval_value, mapxnoise, mapxobservable,
        measurement_transforms, h, sd
    )
end

struct PEtabSDEProblem
    model_info::ModelInfo
    measurements_info::MeasurementsInfo
    sde_solver::SDESolver
    sprob::SDEProblem
    xnames::Vector{Symbol}
    xnames_ps::Vector{Symbol}
    xnominal::ComponentArray{Float64, 1}
    xnominal_ps::ComponentArray{Float64, 1}
end
