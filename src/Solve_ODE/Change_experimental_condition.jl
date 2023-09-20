#=
    Functions for changing odeProblem-parameter to those for an experimental condition
    defined by conditionId
=#


# Change experimental condition when solving the ODE model. A lot of heavy lifting here is done by
# an index which correctly maps parameters for a conditionId to the ODEProblem.
function _changeExperimentalCondition!(pODEProblem::AbstractVector,
                                       u0::AbstractVector,
                                       conditionId::Symbol,
                                       θ_dynamic::AbstractVector,
                                       petab_model::PEtabModel,
                                       θ_indices::ParameterIndices;
                                       computeForwardSensitivites::Bool=false)

    mapConditionId = θ_indices.mapsConiditionId[conditionId]

    # Constant parameters
    pODEProblem[mapConditionId.iODEProblemConstantParameters] .= mapConditionId.constantParameters

    # Parameters to estimate
    pODEProblem[mapConditionId.iODEProblemθDynamic] .= θ_dynamic[mapConditionId.iθDynamic]

    # Given changes in parameters initial values might have to be re-evaluated
    nModelStates = length(petab_model.state_names)
    petab_model.compute_u0!((@view u0[1:nModelStates]), pODEProblem)

    # Account for any potential events (callbacks) which are active at time zero
    for f! in petab_model.check_callback_is_active
        f!(u0, pODEProblem)
    end

    # In case an experimental condition maps directly to the initial value of a state.
    if !isempty(mapConditionId.iODEProblemConstantStates)
        u0[mapConditionId.iODEProblemConstantStates] .= mapConditionId.constantsStates
    end

    # In case we solve the forward sensitivity equations we must adjust the initial sensitives
    # by computing the jacobian at t0
    if computeForwardSensitivites == true
        St0::Matrix{Float64} = Matrix{Float64}(undef, (nModelStates, length(pODEProblem)))
        ForwardDiff.jacobian!(St0, petab_model.compute_u0, pODEProblem)
        u0[(nModelStates+1):end] .= vec(St0)
    end

    return nothing
end


function _changeExperimentalCondition(pODEProblem::AbstractVector,
                                      u0::AbstractVector,
                                      conditionId::Symbol,
                                      θ_dynamic::AbstractVector,
                                      petab_model::PEtabModel,
                                      θ_indices::ParameterIndices)

    mapConditionId = θ_indices.mapsConiditionId[conditionId]

    # For a non-mutating way of mapping constant parameters
    function iConstantParam(iUse::Integer)
        whichIndex = findfirst(x -> x == iUse, mapConditionId.iODEProblemConstantParameters)
        return whichIndex
    end
    # For a non-mutating mapping of parameters to estimate
    function iParametersEst(iUse::Integer)
        whichIndex = findfirst(x -> x == iUse, mapConditionId.iODEProblemθDynamic)
        return mapConditionId.iθDynamic[whichIndex]
    end

    # Constant parameters
    _pODEProblem = [i ∈ mapConditionId.iODEProblemConstantParameters ? mapConditionId.constantParameters[iConstantParam(i)] : pODEProblem[i] for i in eachindex(pODEProblem)]

    # Parameters to estimate
    __pODEProblem = [i ∈ mapConditionId.iODEProblemθDynamic ? θ_dynamic[iParametersEst(i)] : _pODEProblem[i] for i in eachindex(_pODEProblem)]

    # When using AD as Zygote we must use the non-mutating version of evalU0
    _u0 = petab_model.compute_u0(__pODEProblem)

    # In case an experimental condition maps directly to the initial value of a state.
    # Very rare, will fix if ever needed for slow Zygote
    if !isempty(mapConditionId.iODEProblemConstantStates)
        u0[mapConditionId.iODEProblemConstantStates] .= mapConditionId.constantsStates
    end

    # Account for any potential events
    for f! in petab_model.check_callback_is_active
        f!(_u0, __pODEProblem)
    end

    return __pODEProblem, _u0
end
