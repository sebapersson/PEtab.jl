function PEtab.petab_select(path_yaml::String, alg; options = nothing,
                            nmultistarts::Integer = 100, reuse_sensitivities::Bool = false,
                            sampling_method::SamplingAlgorithm = LatinHypercubeSample(),
                            odesolver::Union{Nothing, ODESolver} = nothing,
                            odesolver_gradient::Union{Nothing, ODESolver} = nothing,
                            ss_solver::Union{Nothing, SteadyStateSolver} = nothing,
                            ss_solver_gradient::Union{Nothing, SteadyStateSolver} = nothing,
                            gradient_method::Union{Nothing, Symbol} = nothing,
                            hessian_method::Union{Nothing, Symbol} = nothing,
                            sparse_jacobian::Union{Nothing, Bool} = nothing,
                            sensealg = nothing, sensealg_ss = nothing,
                            chunksize::Union{Nothing, Int64} = nothing,
                            split_over_conditions::Bool = false)
    py"""
    import petab_select

    def get_select_problem(path_yaml):
        return petab_select.Problem.from_yaml(path_yaml)

    def get_iteration_results(select_problem, iteration):
        return petab_select.ui.end_iteration(problem=select_problem,
                                             candidate_space=iteration[petab_select.constants.CANDIDATE_SPACE],
                                             calibrated_models=iteration[petab_select.constants.UNCALIBRATED_MODELS])

    def get_candidate_space(iteration_results, select_problem, first_iteration):
        if first_iteration == True:
            return petab_select.constants.CANDIDATE_SPACE
        else:
            return iteration_results[petab_select.constants.CANDIDATE_SPACE]

    def get_ncandidates(iteration):
        return len(iteration[petab_select.constants.UNCALIBRATED_MODELS])

    def set_criterion_value(select_problem, model, nllh, x):
        model.set_criterion(petab_select.Criterion.NLLH, nllh)
        model.set_criterion(select_problem.criterion,
                            model.get_criterion(select_problem.criterion, compute=True))
        model.estimated_parameters = x
        return

    def get_iteration(select_problem, candidate_space, first_iteration):
        if first_iteration == True:
            return petab_select.ui.start_iteration(problem=select_problem)
        else:
            return petab_select.ui.start_iteration(problem=select_problem,
                                                   candidate_space=candidate_space)

    def get_best_model(select_problem, iteration_results):
        return petab_select.ui.get_best(
            problem=select_problem,
            models=iteration_results[petab_select.constants.CANDIDATE_SPACE].calibrated_models)

    def get_model_info(model):
        return (model.model_subspace_id, model.parameters)

    def write_selection_result(model, path_save):
        model.to_yaml(path_save)

    """

    function _calibrate_candidate(model, select_problem, petab_prob::PEtabODEProblem,
                                  nmultistarts::Integer)::Nothing
        subspace_id, _subspace_parameters = py"get_model_info"(model)
        subspace_parameters = Dict(Symbol(k) => v for (k, v) in pairs(_subspace_parameters))
        @info "Callibrating model $subspace_id"
        prob = PEtab.remake(petab_prob, subspace_parameters)
        if isnothing(options)
            res = PEtab.calibrate_multistart(prob, alg, nmultistarts;
                                             sampling_method = sampling_method)
        else
            res = PEtab.calibrate_multistart(prob, alg, nmultistarts; options = options,
                                             sampling_method = sampling_method)
        end
        xmin_dict = Dict(string.(prob.xnames) .=> res.xmin)
        py"set_criterion_value"(select_problem, model, res.fmin, xmin_dict)
        return nothing
    end

    function calibrate_candidates(iteration, select_problem, petab_prob::PEtabODEProblem,
                                  nmultistarts::Integer)::Nothing
        for model in iteration[py"petab_select.constants.UNCALIBRATED_MODELS"]
            _calibrate_candidate(model, select_problem, petab_prob, nmultistarts)
        end
        return nothing
    end

    # First we use the model-space file to build (from parameter viewpoint) the biggest
    # possible PEtabModel. Then remake is called on the "big" PEtabODEProblem,
    # thus when we compare different models we do not have to pre-compile the model
    # every time (as by default PEtab.jl compiles the model for a given set of parameters
    # to estimate)
    dirmodel = splitdir(path_yaml)[1]
    yaml_file = YAML.load_file(path_yaml)
    model_space_file = CSV.read(joinpath(dirmodel, yaml_file["model_space_files"][1]),
                                DataFrame, stringtype = String)
    xchange = propertynames(model_space_file)[3:end]
    custom_values = Dict()
    [custom_values[xchange[i]] = "estimate" for i in eachindex(xchange)]
    petab_model = PEtabModel(joinpath(dirmodel,
                                      model_space_file[1, :model_subspace_petab_yaml]),
                             build_julia_files = true, verbose = false,
                             write_to_file = false)
    petab_prob = PEtabODEProblem(petab_model; odesolver = odesolver, ss_solver = ss_solver,
                                 odesolver_gradient = odesolver_gradient,
                                 sensealg = sensealg,
                                 ss_solver_gradient = ss_solver_gradient,
                                 chunksize = chunksize, gradient_method = gradient_method,
                                 sensealg_ss = sensealg_ss, hessian_method = hessian_method,
                                 verbose = false, sparse_jacobian = sparse_jacobian,
                                 split_over_conditions = split_over_conditions,
                                 reuse_sensitivities = reuse_sensitivities,
                                 custom_values = custom_values)

    select_problem = py"get_select_problem"(path_yaml)
    str_write = @sprintf("PEtab select problem info\nMethod: %s\nCriterion: %s\n",
                         select_problem.method, select_problem.criterion)
    @info "$str_write"
    k = 1
    local best_model, iteration_results
    while true
        # Start the iterative model selction process
        if k == 1
            iteration = py"get_iteration"(select_problem, nothing, true)
            ncandidates = py"get_ncandidates"(iteration)
            @info "Round $k with $ncandidates candidates - as the code compiles this round \
                   it takes extra long time https://xkcd.com/303/"
        else
            candidate_space = py"get_candidate_space"(iteration_results, select_problem,
                                                      false)
            iteration = py"get_iteration"(select_problem, candidate_space, false)
            ncandidates = py"get_ncandidates"(iteration)
            ncandidates != 0 && @info "Round $k with $ncandidates candidates"
        end
        calibrate_candidates(iteration, select_problem, petab_prob, nmultistarts)
        iteration_results = py"get_iteration_results"(select_problem, iteration)
        ncandidates = py"get_ncandidates"(iteration)
        k += 1
        if ncandidates == 0
            best_model = py"get_best_model"(select_problem, iteration_results)
            break
        end
    end

    # Export best model information to YAML
    path_save = joinpath(splitdir(path_yaml)[1],
                         "PEtab_select_" * string(select_problem.method) * "_" *
                         select_problem.criterion * ".yaml")
    @info "Saving results for best model at $path_save"
    py"write_selection_result"(best_model, path_save)
    return path_save
end
