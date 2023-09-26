# temporary file.

"""
    get_best_ps(res, model::PEtabModel; c_id="__c0__", parmap=true)

From a PEtab result and model, retrives a full paraemter set. 
"""
function get_best_ps(res, model::PEtabModel; c_id="__c0__", parmap=true)    
    ps = first.(model.parameter_map)
    p_vals = last.(model.parameter_map)
    fitted_ps = DataFrame(petab_model.path_parameters)
    condition_ps = DataFrame(petab_model.path_conditions)

    # Loops through all parameters.
    for (p_idx, p) in enumerate(ps)
        if String(ModelingToolkit.getname(p)) in fitted_ps[!, "parameterId"] # If a fitted parameter.
            p_idx = findfirst(isequal(String(ModelingToolkit.getname(p))), fitted_ps[:,1])
            if fitted_ps[p_idx,2] == "lin"
                p_vals[p_idx] = res.xmin[p_idx]
            else # If scale is logarithmic, need to reverse.
                p_vals[p_idx] = 10 ^ res.xmin[p_idx]
            end
        elseif String(ModelingToolkit.getname(p)) in names(condition_ps)[2:end] # If a parameter varrying withg simulation conditions.
            (c_id in condition_ps[:,1]) || error("A condition id was given that could not be found in among the petab model conditions.")
            c_idx = findfirst(isequal(c_id), condition_ps[:,1])
            p_vals[p_idx] = condition_ps[c_idx,String(ModelingToolkit.getname(p))]
        end # If neither, teh value from all_ps should be correct.
    end
    return (parmap ? Pair.(ps, p_vals) : p_vals)
end