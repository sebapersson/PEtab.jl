"""
        ParameterIndices

Parse conditions and build parameter maps for the different parameter types.

There are four types of mechanistic parameters in a PEtab problem:
1. Dynamic (appear in ODE)
2. Noise (appear in noiseParameters columns of measurement table)
3. Observable (appear in observableParameters of measurement table)
4. NonDynamic (do not correspond to any above)

There are three different types of neural-net parameters:
1. inode (appear in the ODE)
2. preode (appear before ODE, setting a subset of parameters values)
3. postode (appear after ODE, and should be treated as NonDynamic, as they basically
    belong to this group).

These parameter types need to be treated separately for computing efficient gradients.
This function extracts which parameter is what type, and builds maps for correctly mapping
the parameter during likelihood computations. It further accounts for parameters potentially
only appearing in a certain simulation conditions.
"""
function ParameterIndices(petab_tables::PEtabTables, paths::Dict{Symbol, String}, sys::ModelSystem, parametermap, speciemap, ml_models::MLModels)::ParameterIndices
    petab_parameters = PEtabParameters(petab_tables[:parameters], petab_tables[:mapping], ml_models)
    petab_ml_parameters = PEtabMLParameters(petab_tables[:parameters], petab_tables[:mapping], ml_models)
    petab_measurements = PEtabMeasurements(petab_tables[:measurements], petab_tables[:observables])
    return ParameterIndices(petab_parameters, petab_ml_parameters, petab_measurements, sys, parametermap, speciemap, petab_tables, paths, ml_models)
end
function ParameterIndices(petab_parameters::PEtabParameters, petab_measurements::PEtabMeasurements, model::PEtabModel)::ParameterIndices
    @unpack speciemap, parametermap, sys_mutated, petab_tables, paths, ml_models = model
    petab_ml_parameters = PEtabMLParameters(petab_tables[:parameters], petab_tables[:mapping], ml_models)
    return ParameterIndices(petab_parameters, petab_ml_parameters, petab_measurements, sys_mutated, parametermap, speciemap, petab_tables, paths, ml_models)
end
function ParameterIndices(petab_parameters::PEtabParameters, petab_ml_parameters::PEtabMLParameters, petab_measurements::PEtabMeasurements, sys::ModelSystem, parametermap, speciemap, petab_tables::PEtabTables, paths::Dict{Symbol, String}, ml_models::Union{Nothing, MLModels})::ParameterIndices
    _check_conditionids(petab_tables, petab_measurements)
    _check_mapping_table(petab_tables, paths, ml_models, petab_parameters, sys)

    conditions_df = petab_tables[:conditions]
    mappings_df = petab_tables[:mapping]

    xids = _get_xids(petab_parameters, petab_ml_parameters, petab_measurements, sys, petab_tables, paths, speciemap, parametermap, ml_models)

    # indices for mapping parameters correctly, e.g. from xest -> xdynamic etc...
    # TODO: SII is going to make this much easier (but the reverse will be harder)
    xindices_est = _get_xindices_xest(xids, ml_models)
    xindices_dynamic = _get_xindices_dynamic(xids, ml_models)
    xindices_notsys = _get_xindices_notsys(xids, ml_models)
    condition_maps = _get_condition_maps(sys, parametermap, speciemap, petab_parameters, conditions_df, mappings_df, xids, ml_models)

    # For each time-point we must build a map that stores if i) noise/obserable parameters
    # are constants, ii) should be estimated, iii) and corresponding index in parameter
    # vector if they should be estimated
    xobservable_maps = _get_map_observable_noise(xids[:observable], petab_measurements,
                                                 petab_parameters; observable = true)
    xnoise_maps = _get_map_observable_noise(xids[:noise], petab_measurements,
                                            petab_parameters; observable = false)
    # If a neural-network sets values for a subset of model parameters, for efficient AD on
    # said network, it is needed to pre-compute the input, pre-allocate the output,
    # and build a map for which parameters in xdynamic the network maps to.
    nn_preode_maps = _get_nn_preode_maps(xids, petab_parameters, petab_tables, paths, ml_models, sys)

    xscale = _get_xscales(xids, petab_parameters)
    _get_xnames_ps!(xids, xscale)
    return ParameterIndices(xindices_est, xids, xindices_notsys, xindices_dynamic, xscale,
                            xobservable_maps, xnoise_maps, condition_maps, nn_preode_maps)
end
