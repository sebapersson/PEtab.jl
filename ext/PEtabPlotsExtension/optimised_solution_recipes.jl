# Plots the optimised solution, and compares it to the data.
@recipe function f(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                   petab_problem::PEtabODEProblem; 
                   observable_ids = [obs.observableId for obs in petab_problem.petab_model.path_observables], 
                   condition_id = [cond.conditionId for cond in petab_problem.petab_model.path_conditions][1])

    # Computes required stuff.
    observable_formulas = [Symbol(obs.observableFormula) for obs in filter(obs -> obs.observableId in observable_ids, petab_problem.petab_model.path_observables)]
    sol = get_odesol(res, petab_problem; condition_id)
    n_obs = length(observable_ids)

    # Get plot options.
    title --> condition_id
    ylabel --> "Concentration"
    seriestype --> reshape([st for _ in 1:n_obs for st in [:scatter :line]], 1, 2n_obs)
    color --> reshape([idx for idx in 1:n_obs for _ in 1:2], 1, 2n_obs)
    label --> reshape(["$(observable_formulas[idx]) ($type)" for idx in 1:n_obs for type in ["measured", "fitted"]], 1, 2n_obs)

    # Get x and y values.
    x_vals = []
    y_vals = []
    for (obs, obs_formula) in zip(observable_ids, observable_formulas)
    # Measurements.
    measurements = filter(m -> (m.simulationConditionId==condition_id) && (m.observableId==obs), petab_problem.petab_model.path_measurements)
    push!(x_vals, [m.time for m in measurements])
    push!(y_vals, [m.measurement for m in measurements])

    # Fitted.
    push!(x_vals, sol.t)
    push!(y_vals, sol[obs_formula])
    end

    # Return output.
    x_vals, y_vals
end

"""
    get_obs_comparison_plots(res, petab_model; kwargs...)

Generates plots comparing the fitted solution to the data. The output is a dict, which contain one entry for each condition_id. Each of these entries contain another dict, each with one entry for each observables_id. Each of these entries contain the output of `plot(res, petab_model; observable_ids=[observable_id], condition_id=condition_id, kwargs...)` for the corresponding condition and observables ids.
"""
function get_obs_comparison_plots(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                                  petab_problem::PEtabODEProblem; kwargs...)
    comparison_dict = Dict()
    for condition_id in [cond.conditionId for cond in petab_problem.petab_model.path_conditions]
        comparison_dict[condition_id] = Dict()
        for observable_id in [obs.observableId for obs in petab_problem.petab_model.path_observables]
            comparison_dict[condition_id][observable_id] = plot(res, petab_problem; observable_ids=[observable_id], condition_id=condition_id, kwargs...)
        end
    end
    return comparison_dict
end

println("HERE")