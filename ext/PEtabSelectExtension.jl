module PEtabSelectExtension

import CSV
using DataFrames: DataFrame
import QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm
using PEtab
import PEtabSelect
import YAML

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
    #=
        Compiling a PEtabODEProblem takes time. Thus, the model-space file is used to
        build (from parameter viewpoint) the biggest possible PEtabModel. Then remake is
        called  on the "big" PEtabODEProblem. Remake does a lot of tricks to not trigger
        a re-compilation.
    =#
    dirmodel = splitdir(path_yaml)[1]
    yaml_file = YAML.load_file(path_yaml)
    path_model_space = joinpath(dirmodel, yaml_file["model_space_files"][1])
    model_space = CSV.read(path_model_space, DataFrame, stringtype = String)
    xchange = propertynames(model_space)[3:end]
    custom_values = Dict(xchange .=> "estimate")
    petab_model = PEtabModel(joinpath(dirmodel, model_space[1, :model_subspace_petab_yaml]))
    petab_prob = PEtabODEProblem(petab_model; odesolver = odesolver, ss_solver = ss_solver, odesolver_gradient = odesolver_gradient, sensealg = sensealg, ss_solver_gradient = ss_solver_gradient, chunksize = chunksize, gradient_method = gradient_method, sensealg_ss = sensealg_ss, hessian_method = hessian_method, sparse_jacobian = sparse_jacobian, split_over_conditions = split_over_conditions, reuse_sensitivities = reuse_sensitivities, custom_values = custom_values)

    select_problem = PEtabSelect.import_problem(path_yaml)
    criterion = split(string(select_problem.criterion), '.')[2]
    method = split(string(select_problem.method), '.')[2]
    @info "Model selection with method $(method) and criteria $(criterion)"

    # Perform model selection until no new candidate models are proposed, then export
    # results for the best model.
    local best_model, iteration_results
    k = 1
    while true
        # Start the iterative model selction process
        if k == 1
            iteration = PEtabSelect.get_iteration_info(select_problem, nothing, true)
            ncandidates = PEtabSelect.get_n_new_models(iteration)
            @info "Round $k with $ncandidates candidates - as the code compiles this round \
                   it takes extra long time https://xkcd.com/303/"
        else
            iteration = PEtabSelect.get_iteration_info(select_problem, iteration_results, false)
            ncandidates = PEtabSelect.get_n_new_models(iteration)
            ncandidates != 0 && @info "Round $k with $ncandidates candidates"
        end
        _calibrate_candidates!(iteration, select_problem, petab_prob, alg, nmultistarts, options, sampling_method)
        iteration_results = PEtabSelect.get_iteration_results(select_problem, iteration)
        ncandidates = PEtabSelect.get_n_new_models(iteration)
        k += 1
        if ncandidates == 0
            best_model = PEtabSelect.get_best_model(select_problem, iteration_results)
            break
        end
    end

    # Export best model information to YAML
    path_save = joinpath(dirmodel, "PEtab_select_$(method)_$(criterion).yaml")
    @info "Saving results for best model at $path_save"
    PEtabSelect.write_model_info(best_model, path_save)
    return path_save
end

function _calibrate_candidates!(iteration, select_problem, petab_prob::PEtabODEProblem, alg, nmultistarts::Integer, options, sampling_method)::Nothing
    uncalibrated_models = PEtabSelect.get_uncalibrated_models(iteration)
    for model in uncalibrated_models
        _calibrate_candidate!(model, select_problem, petab_prob, alg, nmultistarts, options, sampling_method)
    end
    return nothing
end

function _calibrate_candidate!(model, select_problem, petab_prob::PEtabODEProblem, alg, nmultistarts::Integer, options, sampling_method)::Nothing
    subspace_id = PEtabSelect.get_model_subspace_id(model)
    model_parameters = PEtabSelect.get_model_parameters(model)
    @info "Calibrating model $subspace_id"
    prob = PEtab.remake(petab_prob, model_parameters)
    if isnothing(options)
        res = PEtab.calibrate_multistart(prob, alg, nmultistarts; sampling_method = sampling_method)
    else
        res = PEtab.calibrate_multistart(prob, alg, nmultistarts; options = options, sampling_method = sampling_method)
    end
    PEtabSelect.set_criterion!(model, select_problem, res.fmin)
    PEtabSelect.set_parameters!(model, Dict(prob.xnames .=> res.xmin))
    return nothing
end

end
