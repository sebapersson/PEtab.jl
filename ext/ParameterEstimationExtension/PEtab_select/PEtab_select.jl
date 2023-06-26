"""
    runPEtabSelect(pathYAML, optimizer; <keyword arguments>)

Given a PEtab-select YAML file perform model selection with the algorithms specified in the YAML file.

Results are written to a YAML file in the same directory as the PEtab-select YAML file.

Each candidate model produced during the model selection undergoes parameter estimation using local multi-start
optimization. Three optimizers are supported: `optimizer=Fides()` (Fides Newton-trust region), `optimizer=IPNewton()`
from Optim.jl, and `optimizer=LBFGS()` from Optim.jl. Additional keywords for the optimisation are
`nOptimisationStarts::Int`- number of multi-starts for parameter estimation (defaults to 100) and
`optimizationSamplingMethod` - which is any sampling method from QuasiMonteCarlo.jl for generating start guesses
(defaults to LatinHypercubeSample). See also (add callibrate model)

Simulation options can be set using any keyword argument accepted by the `createPEtabODEProblem` function.
For example, setting `gradientMethod=:ForwardDiff` specifies the use of forward-mode automatic differentiation for
gradient computation. If left blank, we automatically select appropriate options based on the size of the problem.
"""
function PEtab.runPEtabSelect(pathYAML::String,
                              optimizer;
                              optimizerOptions=nothing,
                              nOptimisationStarts=100,
                              optimizationSamplingMethod::T=QuasiMonteCarlo.LatinHypercubeSample(),
                              odeSolverOptions::Union{Nothing, ODESolverOptions}=nothing,
                              odeSolverGradientOptions::Union{Nothing, ODESolverOptions}=nothing,
                              ssSolverOptions::Union{Nothing, SteadyStateSolverOptions}=nothing,
                              ssSolverGradientOptions::Union{Nothing, SteadyStateSolverOptions}=nothing,
                              gradientMethod::Union{Nothing, Symbol}=nothing,
                              hessianMethod::Union{Nothing, Symbol}=nothing,
                              sparseJacobian::Union{Nothing, Bool}=nothing,
                              sensealg=nothing,
                              sensealgSS=nothing,
                              chunkSize::Union{Nothing, Int64}=nothing,
                              splitOverConditions::Bool=false,
                              reuseS::Bool=false) where T <: QuasiMonteCarlo.SamplingAlgorithm

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

    def setup_petab_select(pathYAML):
        select_problem = petab_select.Problem.from_yaml(pathYAML)
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


    function _callibrate_model(model, select_problem, _petabProblem::PEtabODEProblem; nOptimisationStarts=nOptimisationStarts)
        subspaceId, subspaceYAML, _subspaceParameters = py"get_model_to_test_info"(model)
        subspaceParameters = Dict(Symbol(k) => v for (k, v) in pairs(_subspaceParameters))
        @info "Callibrating model $subspaceId"
        petabProblem = remakePEtabProblem(_petabProblem, subspaceParameters)
        if isnothing(optimizerOptions)
            _f, _fArg = callibrateModel(petabProblem, optimizer, nOptimisationStarts=nOptimisationStarts, samplingMethod=optimizationSamplingMethod)
        else
            _f, _fArg = callibrateModel(petabProblem, optimizer, nOptimisationStarts=nOptimisationStarts, options=optimizerOptions, samplingMethod=optimizationSamplingMethod)
        end
        whichMin = argmin(_f)
        f = _f[whichMin]
        fArg = _fArg[whichMin, :]

        # Setup dictionary to conveniently storing model parameters
        estimatedParameters = Dict(string(petabProblem.θ_estNames[i]) => fArg[i] for i in eachindex(fArg))
        nDataPoints = length(_petabProblem.computeCost.measurementInfo.measurement)
        py"update_model"(select_problem, model, f, estimatedParameters, nDataPoints)
    end


    function callibrate_candidate_models(candidate_space, select_problem, n_candidates, _petabProblem::PEtabODEProblem; nOptimisationStarts=100)
        for i in 1:n_candidates
            _callibrate_model(candidate_space.models[i], select_problem, _petabProblem::PEtabODEProblem, nOptimisationStarts=nOptimisationStarts)
        end
    end


    # First we use the model-space file to build (from parameter viewpoint) the biggest possible PEtab model. Then remake is called on the "big" petabproblem,
    # thus when we compare different models we do not have to pre-compile the model
    dirModel = splitdir(pathYAML)[1]
    fileYAML = YAML.load_file(pathYAML)
    modelSpaceFile = CSV.File(joinpath(dirModel, fileYAML["model_space_files"][1]), stringtype=String)
    parametersToChange = Symbol.(propertynames(modelSpaceFile)[3:end])
    _customParameterValues = Dict(); [_customParameterValues[parametersToChange[i]] = "estimate" for i in eachindex(parametersToChange)]
    _petabModel = readPEtabModel(joinpath(dirModel, modelSpaceFile[1][:petab_yaml]), forceBuildJuliaFiles=true, verbose=false)
    _petabProblem = createPEtabODEProblem(_petabModel,
                                          odeSolverOptions=odeSolverOptions,
                                          odeSolverGradientOptions=odeSolverGradientOptions,
                                          ssSolverOptions=ssSolverOptions,
                                          ssSolverGradientOptions=ssSolverGradientOptions,
                                          gradientMethod=gradientMethod,
                                          hessianMethod=hessianMethod,
                                          sparseJacobian=sparseJacobian,
                                          sensealg=sensealg,
                                          sensealgSS=sensealgSS,
                                          chunkSize=chunkSize,
                                          splitOverConditions=splitOverConditions,
                                          reuseS=reuseS,
                                          verbose=false,
                                          customParameterValues=_customParameterValues)

    calibrated_models, newly_calibrated_models = py"setup_tracking"()

    select_problem = py"setup_petab_select"(pathYAML)
    strWrite = @sprintf("PEtab select problem info\nMethod: %s\nCriterion: %s\n", select_problem.method, select_problem.criterion)
    @info "$strWrite"

    # Check if there is a predecessor model to setup the parameter space
    candidate_space = py"create_candidate_space"(select_problem)

    k = 1
    n_candidates = py"get_number_of_candidates"(candidate_space)
    local best_model
    while true
        # Start the iterative model selction process
        k == 1 && @info "Model selection round $k with $n_candidates candidates - as the code compiles in this round compiled it takes extra long time https://xkcd.com/303/"
        k != 1 && @info "Model selection round $k with $n_candidates candidates"
        callibrate_candidate_models(candidate_space, select_problem, n_candidates, _petabProblem, nOptimisationStarts=nOptimisationStarts)
        newly_calibrated_models, calibrated_models = py"update_selection"(newly_calibrated_models, calibrated_models, select_problem,  candidate_space)

        py"update_candidate_space"(candidate_space, select_problem, newly_calibrated_models)
        n_candidates = py"get_number_of_candidates"(candidate_space)
        k += 1
        if n_candidates == 0
            best_model = py"get_best_model"(select_problem, calibrated_models)
            break
        end
    end

    # Export best model information to YAML
    pathSave = joinpath(splitdir(pathYAML)[1], "PEtab_select_" * string(select_problem.method) * "_" * select_problem.criterion * ".yaml")
    @info "Saving results for best model at $pathSave"
    py"write_selection_result"(best_model, pathSave)
    return pathSave
end
