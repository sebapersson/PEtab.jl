# Functions used by both the ODE-solvers and PeTab importer.


"""
    setParamToFileValues!(paramMap, stateMap, paramData::ParamData)

    Function that sets the parameter and state values in paramMap and stateMap
    to those in the PeTab parameters file.

    Used when setting up the PeTab cost function, and when solving the ODE-system
    for the values in the parameters-file.
"""
function setParamToFileValues!(paramMap, stateMap, paramData::ParamData)

    parameterNames = paramData.parameterID
    parameterNamesStr = string.([paramMap[i].first for i in eachindex(paramMap)])
    stateNamesStr = replace.(string.([stateMap[i].first for i in eachindex(stateMap)]), "(t)" => "")
    for i in eachindex(parameterNames)

        parameterName = parameterNames[i]
        valChangeTo = paramData.paramVal[i]

        # Check for value to change to in parameter file
        i_param = findfirst(isequal(parameterName), parameterNamesStr)
        i_state = findfirst(isequal(parameterName), stateNamesStr)

        if !isnothing(i_param)
            paramMap[i_param] = Pair(paramMap[i_param].first, valChangeTo)
        elseif !isnothing(i_state)
            stateMap[i_state] = Pair(stateMap[i_state].first, valChangeTo)
        end
    end

end
