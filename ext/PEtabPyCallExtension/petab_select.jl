function PEtab.petab_select(path_yaml::String, alg; options = nothing,
                            nmultistarts = 100, reuse_sensitivities::Bool = false,
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

    def update_model(select_problem, model, nllh, x, ndatapoints):
        model.set_criterion(petab_select.constants.Criterion.NLLH, nllh)
        if select_problem.method == "famos" or select_problem.criterion == "AIC":
            model.set_criterion(petab_select.constants.Criterion.AIC, 2*len(x) + 2*nllh)
        if select_problem.method == "famos" or select_problem.criterion == "BIC":
            model.set_criterion(petab_select.constants.Criterion.BIC, len(x)*np.log(ndatapoints) + 2*nllh)
        if select_problem.method == "famos" or select_problem.criterion == "AICc":
            AIC =  2*len(x) + 2*nllh
            AICc = AIC + ((2*len(x)**2 + 2*len(x)) / (ndatapoints - len(x)-1))
            model.set_criterion(petab_select.constants.Criterion.AICC, AICc)
        model.estimated_parameters = x
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

    function _callibrate_model(model, select_problem, _prob::PEtabODEProblem;
                               nmultistarts = nmultistarts)
        subspace_id, _, _subspace_parameters = py"get_model_to_test_info"(model)
        subspace_parameters = Dict(Symbol(k) => v for (k, v) in pairs(_subspace_parameters))
        @info "Callibrating model $subspace_id"
        prob = remake(_prob, subspace_parameters)
        if isnothing(options)
            _res = PEtab.calibrate_multistart(prob, alg, nmultistarts;
                                              sampling_method = sampling_method)
        else
            _res = PEtab.calibrate_multistart(prob, alg, nmultistarts; options = options,
                                              sampling_method = sampling_method)
        end
        f = _res.fmin
        xmin = _res.xmin
        xmin_dict = Dict(string.(prob.xnames) .=> xmin)
        ndatapoints = nrow(_prob.model_info.model.petab_tables[:measurements])
        py"update_model"(select_problem, model, f, xmin_dict, ndatapoints)
    end

    function callibrate_candidate_models(candidate_space, select_problem, ncandidates,
                                         _prob::PEtabODEProblem; nmultistarts = 100)
        for i in 1:ncandidates
            _callibrate_model(candidate_space.models[i], select_problem,
                              _prob; nmultistarts = nmultistarts)
        end
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
    _model = PEtabModel(joinpath(dirmodel, model_space_file[1, :petab_yaml]),
                        build_julia_files = true, verbose = false, write_to_file = false)
    _prob = PEtabODEProblem(_model; odesolver = odesolver, ss_solver = ss_solver,
                            odesolver_gradient = odesolver_gradient, sensealg = sensealg,
                            ss_solver_gradient = ss_solver_gradient, chunksize = chunksize,
                            gradient_method = gradient_method, sensealg_ss = sensealg_ss,
                            hessian_method = hessian_method, verbose = false,
                            sparse_jacobian = sparse_jacobian,
                            split_over_conditions = split_over_conditions,
                            reuse_sensitivities = reuse_sensitivities,
                            custom_values = custom_values)

    calibrated_models, newly_calibrated_models = py"setup_tracking"()
    select_problem = py"setup_petab_select"(path_yaml)
    str_write = @sprintf("PEtab select problem info\nMethod: %s\nCriterion: %s\n",
                         select_problem.method, select_problem.criterion)
    @info "$str_write"

    # Check if there is a predecessor model to setup the parameter space
    candidate_space = py"create_candidate_space"(select_problem)

    k = 1
    ncandidates = py"get_number_of_candidates"(candidate_space)
    local best_model
    while true
        # Start the iterative model selction process
        k == 1 &&
            @info "Model selection round $k with $ncandidates candidates - as the code \
                   compiles in this round it takes extra long time https://xkcd.com/303/"
        k != 1 && @info "Model selection round $k with $ncandidates candidates"
        callibrate_candidate_models(candidate_space, select_problem, ncandidates,
                                    _prob, nmultistarts = nmultistarts)
        newly_calibrated_models, calibrated_models = py"update_selection"(newly_calibrated_models,
                                                                          calibrated_models,
                                                                          select_problem,
                                                                          candidate_space)

        py"update_candidate_space"(candidate_space, select_problem, newly_calibrated_models)
        ncandidates = py"get_number_of_candidates"(candidate_space)
        k += 1
        if ncandidates == 0
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
