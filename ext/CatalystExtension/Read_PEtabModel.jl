"""
    PEtabModel(system::ReactionSystem,
               simulation_conditions::Dict{String, Dict},
               observables::Dict{String, PEtabObservable},
               measurements::DataFrame,
               petab_parameters::Vector{PEtabParameter};
               state_map::Union{Nothing, Vector{Pair}=nothing,
               parameter_map::Union{Nothing, Vector{Pair}=nothing,
               verbose::Bool=false)::PEtabModel

Create a PEtabModel directly in Julia from a Catalyst reaction system.

For additional information on the input format, see the main documentation. 

# Arguments 
- `system::ReactionSystem`: A Catalyst reaction system.
- `simulation_conditions::Dict{String, T}`: A dictionary specifying values for control parameters/species per simulation condition.
- `observables::Dict{String, PEtab.PEtabObservable}`: A dictionary specifying the observable and noise formulas linking the model to data.
- `measurements::DataFrame`: Measurement data to calibrate the model against.
- `petab_parameters::Vector{PEtab.PEtabParameter}`: Parameters to estimate in PEtabParameter format.
- `state_map=nothing`: An optional state-map to set initial species values to be constant across all simulation conditions.
- `parameter_map=nothing`: An optional state-map to set parameter values to be constant across all simulation conditions.
- `verbose::Bool=false`: Whether to print progress when building the model.

# Example
```julia
# Define a reaction network model 
rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(
    simulation_id=["c0", "c0"],
    obs_id=["obs_a", "obs_a"],
    time=[0, 10.0],
    measurement=[0.7, 0.1],
    noise_parameters=0.5
)

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict())

# PEtab-parameters to estimate
petab_parameters = [
    PEtabParameter(:a0, value=1.0, scale=:lin),
    PEtabParameter(:b0, value=0.0, scale=:lin),
    PEtabParameter(:k1, value=0.8, scale=:lin),
    PEtabParameter(:k2, value=0.6, scale=:lin)
]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem 
petab_model = PEtabModel(
    rn, simulation_conditions, observables, measurements,
    petab_parameters, verbose=false
)
```
"""
function PEtab.PEtabModel(system::ReactionSystem,
                          simulation_conditions::Dict{String, T},
                          observables::Dict{String, PEtab.PEtabObservable},
                          measurements::DataFrame,
                          petab_parameters::Vector{PEtab.PEtabParameter};
                          state_map::Union{Nothing, Vector{Pair{T1, Float64}}}=nothing,
                          parameter_map::Union{Nothing, Vector{Pair{T2, Float64}}}=nothing,
                          verbose::Bool=false)::PEtab.PEtabModel where {T1<:Union{Symbol, Num}, T2<:Union{Symbol, Num}, T<:Dict}

    model_name = "ReactionSystemModel"
    verbose == true && @info "Building PEtabModel for $model_name"

    # Extract model parameters and names
    parameter_names = parameters(system)
    state_names = states(system)

    # Extract relevant PEtab-files, convert to CSV.File
    measurements_data = PEtab.parse_petab_measurements(measurements, observables, simulation_conditions, petab_parameters) |> PEtab.dataframe_to_CSVFile
    parameters_data = PEtab.parse_petab_parameters(petab_parameters, system, simulation_conditions, observables, measurements) |> PEtab.dataframe_to_CSVFile
    observables_data = PEtab.parse_petab_observables(observables) |> PEtab.dataframe_to_CSVFile
    experimental_conditions = PEtab.parse_petab_conditions(simulation_conditions, petab_parameters, system) |> PEtab.dataframe_to_CSVFile
    
    # Build the initial value map (initial values as parameters are set in the reaction system)
    state_map = PEtab.update_state_map(state_map, system, experimental_conditions) # Parameters in condition table
    default_values = Catalyst.get_defaults(system)
    _state_map = [Symbol(replace(string(S), "(t)" => "")) => S ∈ keys(default_values) ? string(default_values[S]) : "0.0" for S in states(system)]
    if !isnothing(state_map)
        state_map_names = [Symbol(_S.first) for _S in state_map]
        for (i, S) in pairs(_state_map)
            if S.first ∉ state_map_names
                continue
            end
            _state_map[i] = _state_map[i].first => string(state_map[findfirst(x -> x == S.first, state_map_names)].second)
        end
    end

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building u0, h and σ functions ...")
    time_taken = @elapsed begin
    h_str, u0!_str, u0_str, σ_str = PEtab.create_σ_h_u0_file(model_name, system, experimental_conditions, measurements_data,
                                                             parameters_data, observables_data, _state_map)
    compute_h = @RuntimeGeneratedFunction(Meta.parse(h_str))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!_str))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0_str))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σ_str))
    end
    verbose == true && @printf(" done. Time = %.1e\n", time_taken)

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u functions ...")
    time_taken = @elapsed begin
    ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = PEtab.create_derivative_σ_h_file(model_name, system, experimental_conditions,
                                                                         measurements_data, parameters_data, observables_data,
                                                                         _state_map)
    compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂u_str))
    compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂p_str))
    compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂u_str))
    compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂p_str))
    end
    verbose == true && @printf(" done. Time = %.1e\n", time_taken)

    # For Callbacks. These function are needed by SBML generated PEtab-files, as for those we as an example rewrite
    # piecewise expressions into events
    write_callbacks_str = "function getCallbacks_" * model_name * "(foo)\n"
    write_tstops_str = "\nfunction compute_tstops(u::AbstractVector, p::AbstractVector)\n"
    write_tstops_str *= "\t return Float64[]\nend\n"
    write_callbacks_str *= "\treturn CallbackSet(), Function[], false\nend"
    get_callback_function = @RuntimeGeneratedFunction(Meta.parse(write_callbacks_str))
    cbset, check_cb_active, convert_tspan = get_callback_function("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    compute_tstops = @RuntimeGeneratedFunction(Meta.parse(write_tstops_str))

    _parameter_map = [Num(p) => 0.0 for p in parameters(system)]
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
                             false,
                             system,
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
                             check_cb_active)
    return petab_model
end


function PEtab.add_model_parameter!(system::ReactionSystem, new_parameter)
    eval(Meta.parse("@parameters " * new_parameter))
    Catalyst.addparam!(system, eval(Meta.parse(new_parameter)))
end
