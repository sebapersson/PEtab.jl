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
function ParameterIndices(petab_tables::Dict{Symbol, DataFrame}, sys, parametermap,
                          speciemap, nn::Union{Nothing, Dict})::ParameterIndices
    petab_parameters = PEtabParameters(petab_tables[:parameters])
    petab_measurements = PEtabMeasurements(petab_tables[:measurements],
                                           petab_tables[:observables])
    return ParameterIndices(petab_parameters, petab_measurements, sys, parametermap,
                            speciemap, petab_tables[:conditions], nn,
                            petab_tables[:mapping_table])
end
function ParameterIndices(petab_parameters::PEtabParameters,
                          petab_measurements::PEtabMeasurements,
                          model::PEtabModel)::ParameterIndices
    @unpack speciemap, parametermap, sys_mutated, petab_tables = model
    return ParameterIndices(petab_parameters, petab_measurements, sys_mutated, parametermap,
                            speciemap, petab_tables[:conditions], model.nn,
                            petab_tables[:mapping_table])
end
function ParameterIndices(petab_parameters::PEtabParameters,
                          petab_measurements::PEtabMeasurements, sys, parametermap,
                          speciemap, conditions_df::DataFrame, nn::Union{Nothing, Dict},
                          mapping_table::Union{Nothing, DataFrame})::ParameterIndices
    _check_conditionids(conditions_df, petab_measurements)
    mapping_table = _check_mapping_table(mapping_table, nn, petab_parameters, sys, conditions_df)

    xids = _get_xids(petab_parameters, petab_measurements, sys, conditions_df, speciemap,
                     parametermap, nn, mapping_table)

    # indices for mapping parameters correctly, e.g. from xest -> xdynamic etc...
    # TODO: SII is going to make this much easier (but the reverse will be harder)
    xindices_est = _get_xindices_xest(xids, nn)
    xindices_dynamic = _get_xindices_dynamic(xids, nn)
    xindices_notsys = _get_xindices_notsys(xids, nn)

    # Maps for mapping to ODEProblem across conditions
    odeproblem_map = _get_odeproblem_map(xids, nn)
    condition_maps = _get_condition_maps(sys, parametermap, speciemap, petab_parameters,
                                         conditions_df, mapping_table, xids)
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
    nn_pre_ode_maps = _get_nn_pre_ode_maps(conditions_df, xids, petab_parameters, mapping_table, nn, sys)

    xscale = _get_xscales(xids, petab_parameters)
    _get_xnames_ps!(xids, xscale)
    return ParameterIndices(xindices_est, xids, xindices_notsys, xindices_dynamic, xscale,
                            xobservable_maps, xnoise_maps, odeproblem_map, condition_maps,
                            nn_pre_ode_maps)
end
