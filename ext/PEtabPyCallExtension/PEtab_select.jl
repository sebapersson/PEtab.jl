function PEtab.run_PEtab_select(path_yaml::String,
                                optimizer;
                                options = nothing,
                                n_multistarts = 100,
                                sampling_method::T = QuasiMonteCarlo.LatinHypercubeSample(),
                                ode_solver::Union{Nothing, ODESolver} = nothing,
                                ode_solver_gradient::Union{Nothing, ODESolver} = nothing,
                                ss_solver::Union{Nothing, SteadyStateSolver} = nothing,
                                ss_solver_gradient::Union{Nothing, SteadyStateSolver} = nothing,
                                gradient_method::Union{Nothing, Symbol} = nothing,
                                hessian_method::Union{Nothing, Symbol} = nothing,
                                sparse_jacobian::Union{Nothing, Bool} = nothing,
                                sensealg = nothing,
                                sensealg_ss = nothing,
                                chunksize::Union{Nothing, Int64} = nothing,
                                split_over_conditions::Bool = false,
                                reuse_sensitivities::Bool = false) where {
                                                                          T <:
                                                                          QuasiMonteCarlo.SamplingAlgorithm
                                                                          }
    py"""
    import petab_select
    import numpy as np

    def print_model(model, select_problem) -> None:
        print(f'''\
              Model subspace ID: {model.model_subspace_id}
              PEtab YAML location: {model.petab_yaml}
              Custom model parameters: {model.parameters}
              Model hash: {model.get_hash()}
              Model ID: {model.model_id}
              Criteria value : {model.get_criterion(select_problem.criterion, compute=False)}
              nllh value : {model.get_criterion("NLLH", compute=False)}
              Estimated parameters : {model.estimated_parameters}
              ''')

    def setup_tracking():
        calibrated_models = {}
        newly_calibrated_models = {}
        return calibrated_models, newly_calibrated_models

    def print_candidate_space(candidate_space, select_problem):
        for candidate in candidate_space.models:
            print_model(candidate, select_problem)

    def get_number_of_candidates(candidate_space):
        return len(candidate_space.models)

    def get_model_to_test_info(model):
        return (model.model_subspace_id, str(model.petab_yaml), model.parameters)

    def setup_petab_select(path_yaml):
        select_problem = petab_select.Problem.from_yaml(path_yaml)
        return select_problem

    def create_candidate_space(select_problem):
        candidate_space = petab_select.ui.candidates(problem=select_problem)
        return candidate_space

    def update_model(select_problem, model, nllh, estimatedParameters, nDataPoints):
        model.set_criterion(petab_select.constants.Criterion.NLLH, nllh)
        if select_problem.method == "famos" or select_problem.criterion == "AIC":
            model.set_criterion(petab_select.constants.Criterion.AIC, 2*len(estimatedParameters) + 2*nllh)
        if select_problem.method == "famos" or select_problem.criterion == "BIC":
            model.set_criterion(petab_select.constants.Criterion.BIC, len(estimatedParameters)*np.log(nDataPoints) + 2*nllh)
        if select_problem.method == "famos" or select_problem.criterion == "AICc":
            AIC =  2*len(estimatedParameters) + 2*nllh
            AICc = AIC + ((2*len(estimatedParameters)**2 + 2*len(estimatedParameters)) / (nDataPoints - len(estimatedParameters)-1))
            model.set_criterion(petab_select.constants.Criterion.AICC, AICc)
        model.estimated_parameters = estimatedParameters
        return

    def update_selection(newly_calibrated_models, calibrated_models, select_problem,  candidate_space):
        newly_calibrated_models = {model.get_hash(): model for model in candidate_space.models}
        calibrated_models.update(newly_calibrated_models)
        select_problem.exclude_models(newly_calibrated_models.values())
        return newly_calibrated_models, calibrated_models

    def update_candidate_space(candidate_space, select_problem, newly_calibrated_models):
        petab_select.ui.candidates(problem=select_problem, candidate_space=candidate_space, newly_calibrated_models=newly_calibrated_models)

    def get_best_model(select_problem, calibrated_models):
        best_model = select_problem.get_best(calibrated_models.values())
        return best_model

    def write_selection_result(model, path_save):
        model.to_yaml(path_save)

    """

    function _callibrate_model(model, select_problem, _petab_problem::PEtabODEProblem;
                               n_multistarts = n_multistarts)
        subspaceId, subspaceYAML, _subspaceParameters = py"get_model_to_test_info"(model)
        subspaceParameters = Dict(Symbol(k) => v for (k, v) in pairs(_subspaceParameters))
        @info "Callibrating model $subspaceId"
        petab_problem = remake_PEtab_problem(_petab_problem, subspaceParameters)
        if isnothing(options)
            _res = PEtab.calibrate_model_multistart(petab_problem, optimizer, n_multistarts,
                                                    nothing,
                                                    sampling_method = sampling_method)
        else
            _res = PEtab.calibrate_model_multistart(petab_problem, optimizer, n_multistarts,
                                                    nothing, options = options,
                                                    sampling_method = sampling_method)
        end
        f = _res.fmin
        fArg = _res.xmin

        # Setup dictionary to conveniently storing model parameters
        estimatedParameters = Dict(string(petab_problem.xnames[i]) => fArg[i]
                                   for i in eachindex(fArg))
        nDataPoints = length(_petab_problem.model.petab_tables[:measurements])
        py"update_model"(select_problem, model, f, estimatedParameters, nDataPoints)
    end

    function callibrate_candidate_models(candidate_space, select_problem, n_candidates,
                                         _petab_problem::PEtabODEProblem;
                                         n_multistarts = 100)
        for i in 1:n_candidates
            _callibrate_model(candidate_space.models[i], select_problem,
                              _petab_problem::PEtabODEProblem,
                              n_multistarts = n_multistarts)
        end
    end

    # First we use the model-space file to build (from parameter viewpoint) the biggest possible PEtab model. Then remake is called on the "big" petabproblem,
    # thus when we compare different models we do not have to pre-compile the model
    dirmodel = splitdir(path_yaml)[1]
    file_yaml = YAML.load_file(path_yaml)
    modelSpaceFile = DataFrame(joinpath(dirmodel, file_yaml["model_space_files"][1]),
                               stringtype = String)
    parametersToChange = Symbol.(propertynames(modelSpaceFile)[3:end])
    _custom_values = Dict()
    [_custom_values[parametersToChange[i]] = "estimate"
     for i in eachindex(parametersToChange)]
    _model = PEtabModel(joinpath(dirmodel, modelSpaceFile[1][:petab_yaml]),
                        build_julia_files = true, verbose = false)
    _petab_problem = PEtabODEProblem(_model,
                                     ode_solver = ode_solver,
                                     ode_solver_gradient = ode_solver_gradient,
                                     ss_solver = ss_solver,
                                     ss_solver_gradient = ss_solver_gradient,
                                     gradient_method = gradient_method,
                                     hessian_method = hessian_method,
                                     sparse_jacobian = sparse_jacobian,
                                     sensealg = sensealg,
                                     sensealg_ss = sensealg_ss,
                                     chunksize = chunksize,
                                     split_over_conditions = split_over_conditions,
                                     reuse_sensitivities = reuse_sensitivities,
                                     verbose = false,
                                     custom_values = _custom_values)

    calibrated_models, newly_calibrated_models = py"setup_tracking"()

    select_problem = py"setup_petab_select"(path_yaml)
    str_write = @sprintf("PEtab select problem info\nMethod: %s\nCriterion: %s\n",
                         select_problem.method, select_problem.criterion)
    @info "$str_write"

    # Check if there is a predecessor model to setup the parameter space
    candidate_space = py"create_candidate_space"(select_problem)

    k = 1
    n_candidates = py"get_number_of_candidates"(candidate_space)
    local best_model
    while true
        # Start the iterative model selction process
        k == 1 &&
            @info "Model selection round $k with $n_candidates candidates - as the code compiles in this round it takes extra long time https://xkcd.com/303/"
        k != 1 && @info "Model selection round $k with $n_candidates candidates"
        callibrate_candidate_models(candidate_space, select_problem, n_candidates,
                                    _petab_problem, n_multistarts = n_multistarts)
        newly_calibrated_models, calibrated_models = py"update_selection"(newly_calibrated_models,
                                                                          calibrated_models,
                                                                          select_problem,
                                                                          candidate_space)

        py"update_candidate_space"(candidate_space, select_problem, newly_calibrated_models)
        n_candidates = py"get_number_of_candidates"(candidate_space)
        k += 1
        if n_candidates == 0
            best_model = py"get_best_model"(select_problem, calibrated_models)
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
