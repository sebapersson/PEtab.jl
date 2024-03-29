function PEtabModel(system::ODESystem,
                    simulation_conditions::Dict,
                    observables::Dict,
                    measurements::DataFrame,
                    petab_parameters::Vector{PEtab.PEtabParameter};
                    state_map::Union{Nothing, AbstractVector} = nothing,
                    parameter_map::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtab.PEtabModel
    model_name = "ODESystemModel"
    return _PEtabModel(system, model_name, simulation_conditions, observables, measurements,
                       petab_parameters, state_map, parameter_map, events, verbose)
end
function PEtabModel(system::ODESystem,
                    observables::Dict,
                    measurements::DataFrame,
                    petab_parameters::Vector{PEtab.PEtabParameter};
                    state_map::Union{Nothing, AbstractVector} = nothing,
                    parameter_map::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtab.PEtabModel
    simulation_conditions = Dict("__c0__" => Dict())
    model_name = "ODESystemModel"
    return _PEtabModel(system, model_name, simulation_conditions, observables, measurements,
                       petab_parameters, state_map, parameter_map, events, verbose)
end
function PEtabModel(system::ReactionSystem,
                    simulation_conditions::Dict,
                    observables::Dict,
                    measurements::DataFrame,
                    petab_parameters::Vector{PEtab.PEtabParameter};
                    state_map::Union{Nothing, AbstractVector} = nothing,
                    parameter_map::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtab.PEtabModel
    model_name = "ReactionSystemModel"
    return PEtab._PEtabModel(system, model_name, simulation_conditions, observables,
                             measurements,
                             petab_parameters, state_map, parameter_map, events, verbose)
end
function PEtabModel(system::ReactionSystem,
                    observables::Dict,
                    measurements::DataFrame,
                    petab_parameters::Vector{PEtab.PEtabParameter};
                    state_map::Union{Nothing, AbstractVector} = nothing,
                    parameter_map::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtab.PEtabModel
    simulation_conditions = Dict("__c0__" => Dict())
    model_name = "ReactionSystemModel"
    return PEtab._PEtabModel(system, model_name, simulation_conditions, observables,
                             measurements,
                             petab_parameters, state_map, parameter_map, events, verbose)
end

function _PEtabModel(system,
                     model_name::String,
                     simulation_conditions::Dict,
                     observables::Dict,
                     measurements::DataFrame,
                     petab_parameters::Vector{PEtab.PEtabParameter},
                     state_map::Union{Nothing, AbstractVector},
                     parameter_map::Union{Nothing, AbstractVector},
                     events::Union{PEtabEvent, AbstractVector, Nothing},
                     verbose::Bool)::PEtab.PEtabModel
    system_mutated = deepcopy(system)
    verbose == true && @info "Building PEtabModel for $model_name"

    # Extract model parameters and names
    parameter_names = parameters(system_mutated)
    state_names = states(system_mutated)

    # Extract relevant PEtab-files, convert to CSV.File
    measurements_data = PEtab.parse_petab_measurements(measurements, observables,
                                                       simulation_conditions,
                                                       petab_parameters) |>
                        PEtab.dataframe_to_CSVFile
    observables_data = PEtab.parse_petab_observables(observables) |>
                       PEtab.dataframe_to_CSVFile

    # Build the initial value map (initial values as parameters are set in the reaction system_mutated)
    default_values = get_default_values(system_mutated)
    _state_map = [Symbol(replace(string(S), "(t)" => "")) => S ∈ keys(default_values) ?
                                                             string(default_values[S]) :
                                                             "0.0"
                  for S in states(system_mutated)]
    add_parameter_inital_values!(system_mutated, _state_map)
    experimental_conditions = PEtab.parse_petab_conditions(simulation_conditions,
                                                           petab_parameters, observables,
                                                           system_mutated) |>
                              PEtab.dataframe_to_CSVFile
    state_map = PEtab.update_state_map(state_map, system_mutated, experimental_conditions) # Parameters in condition table
    if !isnothing(state_map)
        state_map_names = [Symbol(_S.first) for _S in state_map]
        for (i, S) in pairs(_state_map)
            if S.first ∉ state_map_names
                continue
            end
            _state_map[i] = _state_map[i].first => string(state_map[findfirst(x -> x ==
                                                                                   S.first,
                                                                              state_map_names)].second)
        end
    end

    # Once all potential parameters have been added to the system_mutated PEtab parameters can be parsed
    parameters_data = PEtab.parse_petab_parameters(petab_parameters, system_mutated,
                                                   simulation_conditions, observables,
                                                   measurements, state_map,
                                                   parameter_map) |>
                      PEtab.dataframe_to_CSVFile

    verbose == true && printstyled("[ Info:", color = 123, bold = true)
    verbose == true && print(" Building u0, h and σ functions ...")
    time_taken = @elapsed begin
        h_str, u0!_str, u0_str, σ_str = PEtab.create_σ_h_u0_file(model_name, system_mutated,
                                                                 experimental_conditions,
                                                                 measurements_data,
                                                                 parameters_data,
                                                                 observables_data,
                                                                 _state_map)
        compute_h = @RuntimeGeneratedFunction(Meta.parse(h_str))
        compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!_str))
        compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0_str))
        compute_σ = @RuntimeGeneratedFunction(Meta.parse(σ_str))
    end
    verbose == true && @printf(" done. Time = %.1e\n", time_taken)

    verbose == true && printstyled("[ Info:", color = 123, bold = true)
    verbose == true && print(" Building ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u functions ...")
    time_taken = @elapsed begin
        ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = PEtab.create_derivative_σ_h_file(model_name,
                                                                                  system_mutated,
                                                                                  experimental_conditions,
                                                                                  measurements_data,
                                                                                  parameters_data,
                                                                                  observables_data,
                                                                                  _state_map)
        compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂u_str))
        compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂p_str))
        compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂u_str))
        compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂p_str))
    end
    verbose == true && @printf(" done. Time = %.1e\n", time_taken)

    _parameter_map = [Num(p) => 0.0 for p in parameters(system_mutated)]
    for i in eachindex(_parameter_map)
        if isnothing(parameter_map)
            continue
        end
        for j in eachindex(parameter_map)
            if string(_parameter_map[i].first) != string(parameter_map[j].first)
                continue
            end
            _parameter_map[i] = _parameter_map[i].first => parameter_map[j].second
        end
    end

    # For Callbacks. These function are needed by SBML generated PEtab-files, as for those we as an example rewrite
    # piecewise expressions into events
    parameter_info = process_parameters(parameters_data)
    measurement_info = process_measurements(measurements_data, observables_data)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, system_mutated,
                                  _parameter_map, _state_map, experimental_conditions)
    cbset, compute_tstops, convert_tspan = process_petab_events(events, system_mutated,
                                                                θ_indices)

    petab_model = PEtabModel(model_name,
                             compute_h,
                             compute_u0!,
                             compute_u0,
                             compute_σ,
                             compute_∂h∂u!,
                             compute_∂σ∂σu!,
                             compute_∂h∂p!,
                             compute_∂σ∂σp!,
                             compute_tstops,
                             convert_tspan,
                             system,
                             system_mutated,
                             _parameter_map,
                             _state_map,
                             parameter_names,
                             state_names,
                             "",
                             "",
                             measurements_data,
                             experimental_conditions,
                             observables_data,
                             parameters_data,
                             "",
                             "",
                             cbset,
                             Function[],
                             true)
    return petab_model
end

function get_default_values(system::ReactionSystem)
    return Catalyst.get_defaults(system)
end

function add_parameter_inital_values!(system::ReactionSystem, state_map)
    return nothing
end

function add_model_parameter!(system::ReactionSystem, new_parameter)
    eval(Meta.parse("@parameters " * new_parameter))
    Catalyst.addparam!(system, eval(Meta.parse(new_parameter)))
end

function get_default_values(system::ODESystem)
    return ModelingToolkit.get_defaults(system)
end

function PEtab.add_model_parameter!(system::ODESystem, new_parameter)
    eval(Meta.parse("@parameters $new_parameter"))
    push!(ModelingToolkit.get_ps(system),
          eval(Meta.parse("ModelingToolkit.value($new_parameter)")))
end

function add_parameter_inital_values!(system::ODESystem, state_map)

    # Add potential new parameters appearing in the inital conditions to the model. For an ODE-system
    # parameter only appearing in the initial conditions are not a part of the system, and in order
    # to compute gradients accurately they need to be added.
    for i in eachindex(state_map)
        str = replace(String(state_map[i].second), " " => "")
        parameter_names = string.(parameters(system))
        state_names = replace.(string.(states(system)), "(t)" => "")
        i_start, i_end = 1, 1
        char_terminate = ['(', ')', '+', '-', '/', '*', '^']
        while i_end < length(str)
            word_str, i_end = PEtab.get_word(str, i_start, char_terminate)
            i_start = i_start == i_end ? i_end + 1 : i_end

            if isempty(word_str)
                continue
            end
            if PEtab.is_number(word_str)
                continue
            end
            if word_str ∈ parameter_names
                continue
            end
            if word_str ∈ state_names
                str_write = "Initial default value of a state cannot be another state, as currently for $word_str"
                throw(PEtab.PEtabFormatError("$str_write"))
            end
            # At this point the initial default value is a parameter, that does not occur in the ODESystem. In order
            # to ensure correct comuations downstream it must be added as a parameter
            PEtab.add_model_parameter!(system, word_str)
        end
    end
    return
end
