# Plots the optimised solution, and compares it to the data.
@recipe function f(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult},
                   petab_problem::PEtabODEProblem;
                   observable_ids = [obs.observableId
                                     for obs in petab_problem.petab_model.petab_tables[:observables]],
                   condition_id = [cond.conditionId
                                   for cond in petab_problem.petab_model.petab_tables[:conditions]][1])

    # Get plot options.
    title --> condition_id
    ylabel --> "Concentration"

    # Prepares empty vectors with required plot inputs.
    seriestype = []
    color = []
    label = []
    x_vals = []
    y_vals = []

    # Loops through all observables, computing the required plot inputs.
    all_obs = petab_problem.petab_model.petab_tables[:observables]
    for (obs_idx, obs_id) in enumerate(observable_ids)
        t_observed, h_observed, label_observed, t_model, h_model, label_model, smooth_sol = _get_observable(res.xmin,
                                                                                                            petab_problem,
                                                                                                            condition_id,
                                                                                                            obs_id)

        # Plot args.
        append!(seriestype, [:scatter, smooth_sol ? :line : :scatter])
        append!(color, [obs_idx, obs_idx])
        obs_formula = all_obs[all_obs.observableId .== observable_ids[1], :][1,
                                                                             1].observableFormula
        append!(label, ["$(obs_formula) ($type)" for type in ["measured", "fitted"]])

        # Measured plot values.
        push!(x_vals, t_observed)
        push!(y_vals, h_observed)

        # Fitted plot values.
        push!(x_vals, t_model)
        push!(y_vals, h_model)
    end

    # Set reshaped plot arguments
    n_obs = length(observable_ids)
    seriestype --> reshape(seriestype, 1, 2n_obs)
    color --> reshape(color, 1, 2n_obs)
    label --> reshape(label, 1, 2n_obs)

    # Return output.
    x_vals, y_vals
end

"""
    get_obs_comparison_plots(res, petab_model; kwargs...)

Generates plots comparing the fitted solution to the data. The output is a dict, which contain one entry for each condition_id. Each of these entries contain another dict, each with one entry for each observables_id. Each of these entries contain the output of `plot(res, petab_model; observable_ids=[observable_id], condition_id=condition_id, kwargs...)` for the corresponding condition and observables ids.
"""
function PEtab.get_obs_comparison_plots(res::Union{PEtabOptimisationResult,
                                                   PEtabMultistartOptimisationResult},
                                        petab_problem::PEtabODEProblem; kwargs...)
    comparison_dict = Dict()
    for condition_id in [cond.conditionId
                         for cond in petab_problem.petab_model.petab_tables[:conditions]]
        comparison_dict[condition_id] = Dict()
        for observable_id in [obs.observableId
                              for obs in petab_problem.petab_model.petab_tables[:observables]]
            comparison_dict[condition_id][observable_id] = plot(res, petab_problem;
                                                                observable_ids = [observable_id],
                                                                condition_id = condition_id,
                                                                kwargs...)
        end
    end
    return comparison_dict
end

"""
    _get_observable(θ::Vector{Float64}, petab_problem::PEtabODEProblem,
                    condition_id::String, observable_id::String)

Return the model values for a given observable_id and condition_id using the parameter vector θ.

This function is primarily used for plotting purposes. It is important to note that for a single condition_id and
observable_id, the function may return simulation results corresponding to multiple conditions if the model has
different pre-equilibrium ids.

# Returns
- `t_observed::Vector{Float64}`: Time points for the observed data (x-axis).
- `h_observed::Vector{Float64}`: Observed data values corresponding to these time points.
- `label_observed::Vector{String}`: Labels denoting the condition id, considering any pre-equilibrium scenarios.
- `t_model::Vector{Float64}`: Time points for the model data (x-axis).
- `h_model::Vector{Float64}`: Model's predicted values corresponding to these time points.
- `label_model::Vector{String}`: Model labels denoting the condition id, considering any pre-equilibrium scenarios.
- `smooth_sol::Bool`: Indicates whether the returned solution is smooth, i.e., there are no
    observable parameters.
"""
function _get_observable(θ::Vector{Float64}, petab_problem::PEtabODEProblem,
                         condition_id::String, observable_id::String)

    # All measurment data is stored in petab_problem.petab_model.petab_tables[:measurements]
    measurement_data = petab_problem.petab_model.petab_tables[:measurements] |> DataFrame
    condition_ids = measurement_data[!, :simulationConditionId]
    observable_ids = measurement_data[!, :observableId]

    @assert condition_id in condition_ids "$condition_id in one of the model's simulations ids"
    @assert observable_id in observable_ids "$observable_id in one of the model's observable ids"

    # Identify which data-points in measurment data we should plot simulations for
    idata = findall(condition_ids .== condition_id .&& observable_ids .== observable_id)
    if isempty(idata)
        return Float64[], Float64[], String[], Float64[], Float64[], String[], true
    end

    # Extract measurement value and observed time for the data. Alongside create an id-tag.
    # Notice that an observable_id for a condition_id can have in a sense different simulation
    # values due to a potential pre-eq id, different pre-eq should be labelled accordingly in
    # the plot
    pre_eq_ids = petab_problem.measurement_info.conditionids[:pre_equilibration][idata]
    t_observed = measurement_data[idata, :time]
    h_observed = measurement_data[idata, :measurement]
    label_observed = [pre_eq_id == :None ? condition_id :
                      "pre_" * string(pre_eq_id) * "_" * condition_id
                      for pre_eq_id in pre_eq_ids]

    t_model = Float64[]
    h_model = Float64[]
    label_model = String[]
    smooth_sol::Bool = true
    for pre_eq_id in unique(pre_eq_ids)
        if pre_eq_id == :None
            _idata = idata
            sol = get_odesol(θ, petab_problem; condition_id = condition_id)
        else
            _idata = idata[findall(pre_eq_ids .== pre_eq_id)]
            sol = get_odesol(θ, petab_problem; condition_id = condition_id,
                             pre_eq_id = pre_eq_id)
        end

        # If number of observable parameters is non-zero it does not make sense to return a
        # a smooth trajectory, but if they are zero returning a smooth trajectory is preferred
        # as it looks better
        mapxobservables = petab_problem.θ_indices.mapxobservable[_idata]
        smooth_sol = all([map.n_parameters == 0 for map in mapxobservables])
        # For smooth trajectory must solve ODE and compute the observable function
        if smooth_sol == true
            @unpack θ_indices, cache, measurement_info, parameter_info, petab_model = petab_problem
            xdynamic, xobservable, xnoise, xnondynamic = PEtab.split_x(θ, θ_indices)
            xobservable_ps = PEtab.transform_x(xobservable, θ_indices.xids[:observable],
                                             θ_indices, :xobservable, cache)
            xnondynamic_ps = PEtab.transform_x(xnondynamic, θ_indices.xids[:nondynamic],
                                              θ_indices, :xnondynamic, cache)
            i_measurement = _idata[1]
            _h_model = similar(sol.t)
            for i in eachindex(_h_model)
                u = sol[:, i]
                _h_model[i] = PEtab.computeh(u, sol.t[i], sol.prob.p, xobservable_ps,
                                             xnondynamic_ps, petab_model,
                                             i_measurement, measurement_info, θ_indices,
                                             parameter_info)
            end
            _t_model = sol.t

        else
            _t_model = measurement_data[_idata, :time]
            _h_model = petab_problem.compute_simulated_values(θ)[_idata]
        end

        h_model = vcat(h_model, _h_model)
        t_model = vcat(t_model, _t_model)
        _label = Vector{String}(undef, length(_h_model))
        if pre_eq_id == :None
            _label .= condition_id
        else
            _label .= "pre_" * string(pre_eq_id) * "_" * condition_id
        end
        label_model = vcat(label_model, _label)
    end

    return t_observed, h_observed, label_observed, t_model, h_model, label_model, smooth_sol
end
