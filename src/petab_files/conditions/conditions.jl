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
function ParameterIndices(petab_tables::Dict{Symbol, DataFrame}, sys::ModelSystem, parametermap, speciemap, nnmodels::Union{Nothing, Dict{Symbol, <:NNModel}})::ParameterIndices
    petab_parameters = PEtabParameters(petab_tables[:parameters], petab_tables[:mapping_table], nnmodels)
    petab_net_parameters = PEtabNetParameters(petab_tables[:parameters], petab_tables[:mapping_table], nnmodels)
    petab_measurements = PEtabMeasurements(petab_tables[:measurements], petab_tables[:observables])
    return ParameterIndices(petab_parameters, petab_net_parameters, petab_measurements, sys, parametermap, speciemap, petab_tables, nnmodels)
end
function ParameterIndices(petab_parameters::PEtabParameters, petab_measurements::PEtabMeasurements, model::PEtabModel)::ParameterIndices
    @unpack speciemap, parametermap, sys_mutated, petab_tables, nnmodels = model
    petab_net_parameters = PEtabNetParameters(petab_tables[:parameters], petab_tables[:mapping_table], nnmodels)
    return ParameterIndices(petab_parameters, petab_net_parameters, petab_measurements, sys_mutated, parametermap, speciemap, petab_tables, nnmodels)
end
function ParameterIndices(petab_parameters::PEtabParameters, petab_net_parameters::PEtabNetParameters, petab_measurements::PEtabMeasurements, sys::ModelSystem, parametermap, speciemap, petab_tables::Dict{Symbol, DataFrame}, nnmodels::Union{Nothing, Dict{Symbol, <:NNModel}})::ParameterIndices
    _check_conditionids(petab_tables, petab_measurements)
    _check_mapping_table(petab_tables, nnmodels, petab_parameters, sys)

    xids = _get_xids(petab_parameters, petab_net_parameters, petab_measurements, sys, petab_tables, speciemap, parametermap, nnmodels)

    # indices for mapping parameters correctly, e.g. from xest -> xdynamic etc...
    # TODO: SII is going to make this much easier (but the reverse will be harder)
    xindices_est = _get_xindices_xest(xids, nnmodels)
    xindices_dynamic = _get_xindices_dynamic(xids, nnmodels)
    xindices_notsys = _get_xindices_notsys(xids, nnmodels)

    # Maps for mapping to ODEProblem across conditions
    odeproblem_map = _get_odeproblem_map(xids, nnmodels)
    condition_maps = _get_condition_maps(sys, parametermap, speciemap, petab_parameters, petab_tables, xids)
    # For each time-point we must build a map that stores if i) noise/obserable parameters
    # are constants, ii) should be estimated, iii) and corresponding index in parameter
    # vector if they should be estimated
    xobservable_maps = _get_map_observable_noise(xids[:observable], petab_measurements,
                                                 petab_parameters; observable = true)
    xnoise_maps = _get_map_observable_noise(xids[:noise], petab_measurements,
                                            petab_parameters; observable = false)
    # If a neural-network sets values for a subset of model parameters, for efficent AD on
    # said network, it is neccesary to pre-compute the input, pre-allocate the output,
    # and build a map for which parameters in xdynamic the network maps to.
    nn_preode_maps = _get_nn_preode_maps(xids, petab_parameters, petab_tables, nnmodels, sys)

    xscale = _get_xscales(xids, petab_parameters)
    _get_xnames_ps!(xids, xscale)
    return ParameterIndices(xindices_est, xids, xindices_notsys, xindices_dynamic, xscale,
                            xobservable_maps, xnoise_maps, odeproblem_map, condition_maps,
                            nn_preode_maps)
end
