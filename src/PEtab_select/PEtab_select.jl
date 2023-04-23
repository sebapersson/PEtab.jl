using PEtab
using PyCall
using Printf
using OrdinaryDiffEq
using Optim

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

    def get_model_to_test(candidate_space, index):
        model = candidate_space.models[index]
        return (model.model_subspace_id, str(model.petab_yaml), model.parameters)

    def setup_petab_select(pathYAML):
        select_problem = petab_select.Problem.from_yaml(pathYAML)
        return select_problem

    def create_candidate_space(select_problem):
        candidate_space = petab_select.ui.candidates(problem=select_problem)
        return candidate_space

    def update_model(select_problem, candidate_space, nllh, estimatedParameters, nDataPoints, index):
        model = candidate_space.models[index]
        model.set_criterion(petab_select.constants.Criterion.NLLH, nllh)
        if select_problem.criterion == "AIC":
            model.set_criterion(select_problem.criterion, 2*len(estimatedParameters) + 2*nllh)
        elif select_problem.criterion == "BIC":
            model.set_criterion(select_problem.criterion, len(estimatedParameters)*np.log(nDataPoints) + 2*nllh)
        elif select_problem.criterion == "AICc":
            AIC =  2*len(estimatedParameters) + 2*nllh
            AICc = AIC + ((2*len(estimatedParameters)**2 + 2*len(estimatedParameters)) / (nDataPoints - len(estimatedParameters)-1))
            model.set_criterion(select_problem.criterion, AICc)
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

    function callibrate_models(candidate_space, select_problem, n_candidates; nStartGuesses=100)
        for i in 1:n_candidates
            subspaceId, subspaceYAML, _subspaceParameters = py"get_model_to_test"(candidate_space, i-1)
            subspaceParameters = Dict(Symbol(k) => v for (k, v) in pairs(_subspaceParameters)) 

            @info "Callibrating model $subspaceId"
            petabModel = readPEtabModel(subspaceYAML, forceBuildJuliaFiles=true, customParameterValues=subspaceParameters, verbose=false)
            petabProblem = setupPEtabODEProblem(petabModel, 
                                                odeSolverOptions, 
                                                gradientMethod=gradientMethod,
                                                hessianMethod=hessianMethod,
                                                reuseS=reuseS,
                                                sensealg=sensealg,
                                                customParameterValues=subspaceParameters, verbose=false)
            if isnothing(optimizerOptions)
                f, fArg = callibrateModel(petabProblem, optimizer, nStartGuesses=nStartGuesses)
            else
                f, fArg = callibrateModel(petabProblem, optimizer, nStartGuesses=nStartGuesses, options=optimizerOptions)
            end
            # Setup dictionary to conveniently storing model parameters 
            estimatedParameters = Dict(string(petabProblem.θ_estNames[i]) => fArg[i] for i in eachindex(fArg)) 
            nDataPoints = length(petabProblem.computeCost.measurementInfo.measurement)
            py"update_model"(select_problem, candidate_space, f, estimatedParameters, nDataPoints, i-1)
        end
        return nothing, false
    end

    calibrated_models, newly_calibrated_models = py"setup_tracking"()

    select_problem = py"setup_petab_select"(pathYAML)
    strWrite = @sprintf("PEtab select problem info\nMethod: %s\nCriterion: %s\n", select_problem.method, select_problem.criterion)
    @info "$strWrite"

    candidate_space = py"create_candidate_space"(select_problem)

    k = 1
    n_candidates = py"get_number_of_candidates"(candidate_space)
    local best_model
    while true
        # Start the iterative model selction process 
        @info "Model selection round $k with $n_candidates candidates"
        callibrate_models(candidate_space, select_problem, n_candidates, nStartGuesses=nMultiStarts)
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

# TODO : 
# 1. Setup callibration with Fides. Done 
# 2. Run all tests. Done 
# 3. Try estimating blasi model at reported values. Done 
# 4. Launch bigg on cluster. TODO 
# 5. Launch Fabian adjoint 
# 6. Complete fun presentation 
# 7. Setup notebook second order 
# 8. Wrap Ipopt 
# 9. Investigate gradient fixing parameters (should be doable)
# 10. Work on benchmark paper 
