#=
    Functions for changing ode_problem-parameter to those for an experimental condition
    defined by condition_id
=#

# Change experimental condition when solving the ODE model. A lot of heavy lifting here is done by
# an index which correctly maps parameters for a condition_id to the ODEProblem.
function _change_simulation_condition!(p_ode_problem::AbstractVector,
                                       u0::AbstractVector,
                                       condition_id::Symbol,
                                       θ_dynamic::AbstractVector,
                                       petab_model::PEtabModel,
                                       θ_indices::ParameterIndices;
                                       compute_forward_sensitivites::Bool = false)::Nothing
    map_condition_id = θ_indices.maps_conidition_id[condition_id]

    # Constant parameters
    p_ode_problem[map_condition_id.isys_constant_values] .= map_condition_id.constant_values

    # Parameters to estimate
    p_ode_problem[map_condition_id.ix_sys] .= θ_dynamic[map_condition_id.ix_dynamic]

    # Given changes in parameters initial values might have to be re-evaluated
    n_model_states = length(petab_model.state_names)
    petab_model.compute_u0!((@view u0[1:n_model_states]), p_ode_problem)

    # Account for any potential events (callbacks) which are active at time zero
    for f! in petab_model.check_callback_is_active
        f!(u0, p_ode_problem)
    end

    # In case we solve the forward sensitivity equations we must adjust the initial sensitives
    # by computing the jacobian at t0
    if compute_forward_sensitivites == true
        S_t0::Matrix{Float64} = Matrix{Float64}(undef,
                                                (n_model_states, length(p_ode_problem)))
        ForwardDiff.jacobian!(S_t0, petab_model.compute_u0, p_ode_problem)
        u0[(n_model_states + 1):end] .= vec(S_t0)
    end

    return nothing
end

function _change_simulation_condition(p_ode_problem::AbstractVector,
                                      u0::AbstractVector,
                                      condition_id::Symbol,
                                      θ_dynamic::AbstractVector,
                                      petab_model::PEtabModel,
                                      θ_indices::ParameterIndices)
    map_condition_id = θ_indices.maps_conidition_id[condition_id]

    # For a non-mutating way of mapping constant parameters
    function i_constant_param(i_use::Integer)
        which_index = findfirst(x -> x == i_use, map_condition_id.isys_constant_values)
        return which_index
    end
    # For a non-mutating mapping of parameters to estimate
    function i_parameters_est(i_use::Integer)
        which_index = findfirst(x -> x == i_use, map_condition_id.ix_sys)
        return map_condition_id.ix_dynamic[which_index]
    end

    # Constant parameters
    _p_ode_problem = [i ∈ map_condition_id.isys_constant_values ?
                      map_condition_id.constant_values[i_constant_param(i)] :
                      p_ode_problem[i] for i in eachindex(p_ode_problem)]

    # Parameters to estimate
    __p_ode_problem = [i ∈ map_condition_id.ix_sys ?
                       θ_dynamic[i_parameters_est(i)] : _p_ode_problem[i]
                       for i in eachindex(_p_ode_problem)]

    # When using AD as Zygote we must use the non-mutating version of evalU0
    _u0 = petab_model.compute_u0(__p_ode_problem)

    # In case an experimental condition maps directly to the initial value of a state.
    # Very rare, will fix if ever needed for slow Zygote
    if !isempty(map_condition_id.i_ode_constant_states)
        u0[map_condition_id.i_ode_constant_states] .= map_condition_id.constant_states
    end

    # Account for any potential events
    for f! in petab_model.check_callback_is_active
        f!(_u0, __p_ode_problem)
    end

    return __p_ode_problem, _u0
end
