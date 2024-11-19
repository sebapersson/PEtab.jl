function _check_conditionids(conditions_df::DataFrame,
                             petab_measurements::PEtabMeasurements)::Nothing
    ncol(conditions_df) == 1 && return nothing
    @unpack pre_equilibration_condition_id, simulation_condition_id = petab_measurements
    measurementids = unique(vcat(pre_equilibration_condition_id, simulation_condition_id))
    for conditionid in (conditions_df[!, :conditionId] .|> Symbol)
        conditionid in measurementids && continue
        @warn "Simulation condition id $conditionid does not appear in the measurements " *
              "table. Therefore no measurement corresponds to this id."
    end
    return nothing
end

function _check_mapping_table(mapping_table::Union{DataFrame, Nothing}, nn::Union{Nothing, Dict}, petab_parameters::PEtabParameters, sys::ModelSystem, conditions_df::DataFrame)::DataFrame
    isempty(mapping_table) && return DataFrame()
    isnothing(nn) && return DataFrame()
    state_ids = _get_state_ids(sys) .|> Symbol
    xids_sys = _get_xids_sys(sys)
    model_variables = Iterators.flatten((state_ids, xids_sys, petab_parameters.parameter_id))

    # Sanity check ioId column
    pattern = r"^(input\d|output\d)$"
    for io_id in string.(mapping_table[!, :ioId])
        if !occursin(pattern, io_id)
            throw(PEtabInputError("In mapping table, in ioId column allowed values are \
                                   only input{:digit} or output{:digit} where digit is \
                                   the number of the input/output to the network. Not \
                                   $io_id"))
        end
    end
    # Sanity check ioValue column (input and outputs to neural-net)
    for netid in Symbol.(unique(mapping_table[!, :netId]))
        if !haskey(nn, netid)
            throw(PEtabInputError("Neural network id $netid provided in the mapping table \
                                   does not correspond to any Neural Network id provided \
                                   in the PEtab problem"))
        end
        outputs = _get_net_values(mapping_table, netid, :outputs) .|> Symbol
        inputs = _get_net_values(mapping_table, netid, :inputs) .|> Symbol
        input_variables = _get_nn_input_variables(inputs, conditions_df, petab_parameters, sys)

        # If input_variables is empty all inputs are numbers which can always be handled
        if isempty(input_variables)
            continue
        # If all outputs maps to ODEProblem parameters, it is a pre-nn ODE case. In this
        # case all input variables must be PEtab parameters (or numbers which are already
        # filtered out from input_variables)
        elseif all([output in xids_sys for output in outputs])
            if !all([ipv in petab_parameters.parameter_id for ipv in input_variables])
                throw(PEtabInputError("If mapping table output is ODEProblem parameters \
                                       input must be a PEtabParameter, this does not hold \
                                       for $inputs"))
            end
            continue
        # If all inputs maps to model variables, the nn must be in the observable formula,
        # else something is wrong
        elseif !all([ipv in model_variables for ipv in input_variables])
            throw(PEtabInputError("If mapping table output is a parameter in the \
                                   observable/sd formula, input must be a PEtabParameter, \
                                   model specie, or model parameter. Does not hold for \
                                   all inputs in $inputs"))
            continue
        end
    end
    return DataFrame(netId = Symbol.(mapping_table[!, :netId]),
                     ioId = Symbol.(mapping_table[!, :ioId]),
                     ioValue = Symbol.(mapping_table[!, :ioValue]))
end
