using PEtab
using PyCall
using Printf
using OrdinaryDiffEq
using Optim
using DataFrames
using CSV
using YAML

# Adding functionality for working with PEtab select
# Inputs 
#    pathYAML - path to the PEtab select YAML file 
#    pathState - path where the state of the problem is stored 
#    pathOutput - path to where the output file is stored 
#    method - method used to find the best model as a symbol 

pathPythonExe = joinpath("/", "home", "sebpe", "anaconda3", "envs", "PeTab", "bin", "python")
ENV["PYTHON"] = pathPythonExe
import Pkg; Pkg.build("PyCall")


function call_py(pathYAML::String, 
                 optimizer; 
                 optimizerOptions=nothing,
                 odeSolverOptions=getODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8), 
                 gradientMethod=:ForwardDiff, 
                 hessianMethod=:ForwardDiff, 
                 sensealg=:ForwardDiff,
                 reuseS=false, 
                 nMultiStarts=5)

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

    def get_predecessor_model(path_yaml):
        model = petab_select.models_from_yaml_list(path_yaml)
        return model[0]

    def setup_petab_select(pathYAML):
        select_problem = petab_select.Problem.from_yaml(pathYAML)
        return select_problem

    def create_candidate_space(select_problem, start_model=None):
        if start_model == None:
            candidate_space = petab_select.ui.candidates(problem=select_problem)
        else:
            print("Got here :)")
            candidate_space = petab_select.ui.candidates(problem=select_problem, previous_predecessor_model=start_model)
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


    function _callibrate_model(model, select_problem, _petabProblem::PEtabODEProblem; nStartGuesses=nStartGuesses)
        subspaceId, subspaceYAML, _subspaceParameters = py"get_model_to_test_info"(model)
        subspaceParameters = Dict(Symbol(k) => v for (k, v) in pairs(_subspaceParameters)) 
        @info "Callibrating model $subspaceId"
        petabProblem = remakePEtabProblem(_petabProblem, subspaceParameters)
        if isnothing(optimizerOptions)
            f, fArg = callibrateModel(petabProblem, optimizer, nStartGuesses=nStartGuesses)
        else
            f, fArg = callibrateModel(petabProblem, optimizer, nStartGuesses=nStartGuesses, options=optimizerOptions)
        end
        # Setup dictionary to conveniently storing model parameters 
        estimatedParameters = Dict(string(petabProblem.θ_estNames[i]) => fArg[i] for i in eachindex(fArg)) 
        nDataPoints = length(_petabProblem.computeCost.measurementInfo.measurement)
        py"update_model"(select_problem, model, f, estimatedParameters, nDataPoints)
        return model
    end


    function callibrate_predecessor_model(pathYAML, select_problem, _petabProblem::PEtabODEProblem; nStartGuesses=100)
        _model = py"get_predecessor_model"(pathYAML)
        model = _callibrate_model(_model, select_problem, _petabProblem::PEtabODEProblem, nStartGuesses=nStartGuesses)
        return model
    end


    function callibrate_candidate_models(candidate_space, select_problem, n_candidates, _petabProblem::PEtabODEProblem; nStartGuesses=100)
        for i in 1:n_candidates
            _ = _callibrate_model(candidate_space.models[i], select_problem, _petabProblem::PEtabODEProblem, nStartGuesses=nStartGuesses)
        end
        return nothing, false
    end

    # First we use the model-space file to build (from parameter viewpoint) the biggest possible PEtab model. Then remake is called on the "big" petabproblem, 
    # thus when we compare different models we do not have to pre-compile the model 
    dirModel = splitdir(pathYAML)[1]
    fileYAML = YAML.load_file(pathYAML)
    modelSpaceFile = CSV.read(joinpath(dirModel, fileYAML["model_space_files"][1]), DataFrame)
    parametersToChange = Symbol.(names(modelSpaceFile)[3:end])
    _customParameterValues = Dict(); [_customParameterValues[parametersToChange[i]] = "estimate" for i in eachindex(parametersToChange)]
    _petabModel = readPEtabModel(joinpath(dirModel, modelSpaceFile[1, :petab_yaml]), forceBuildJuliaFiles=true, verbose=false)
    _petabProblem = setupPEtabODEProblem(_petabModel, odeSolverOptions, 
                                         gradientMethod=gradientMethod, 
                                         hessianMethod=hessianMethod, 
                                         sensealg=sensealg,
                                         reuseS=reuseS, 
                                         verbose=false, 
                                         customParameterValues=_customParameterValues)
    
    calibrated_models, newly_calibrated_models = py"setup_tracking"()

    select_problem = py"setup_petab_select"(pathYAML)
    strWrite = @sprintf("PEtab select problem info\nMethod: %s\nCriterion: %s\n", select_problem.method, select_problem.criterion)
    @info "$strWrite"

    # Check if there is a predecessor model to setup the parameter space 
    if "candidate_space_arguments" ∈ keys(fileYAML) && "predecessor_model" ∈ keys(fileYAML["candidate_space_arguments"])
        path_predecessor = joinpath(dirname(pathYAML), fileYAML["candidate_space_arguments"]["predecessor_model"])
        start_model = callibrate_predecessor_model(path_predecessor, select_problem, _petabProblem, nStartGuesses=nMultiStarts)
        candidate_space = py"create_candidate_space"(select_problem, start_model=start_model)
    else
        candidate_space = py"create_candidate_space"(select_problem)
    end

    k = 1
    n_candidates = py"get_number_of_candidates"(candidate_space)
    local best_model
    while true
        # Start the iterative model selction process 
        k == 1 && @info "Model selection round $k with $n_candidates candidates - as the code compiles in this round compiled it takes extra long time https://xkcd.com/303/"
        k != 1 && @info "Model selection round $k with $n_candidates candidates"
        callibrate_candidate_models(candidate_space, select_problem, n_candidates, _petabProblem, nStartGuesses=nMultiStarts)
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
end


pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0001", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0002", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0003", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0004", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0005", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0006", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0007", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0008", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff)

pathYAML = joinpath(@__DIR__, "..", "..", "test_local", "PEtab_select", "0009", "petab_select_problem.yaml")
call_py(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nMultiStarts=100)
