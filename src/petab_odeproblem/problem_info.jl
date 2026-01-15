const GRADIENT_METHODS = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint]
const HESSIAN_METHODS = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
const FIM_METHODS = [nothing, :ForwardDiff, :GaussNewton]

function PEtabODEProblemInfo(model::PEtabModel, model_info::ModelInfo, odesolver,
                             odesolver_gradient, ss_solver, ss_solver_gradient,
                             gradient_method, hessian_method, FIM_method, sensealg,
                             sensealg_ss, reuse_sensitivities::Bool, sparse_jacobian,
                             specialize_level, chunksize, split_over_conditions,
                             verbose::Bool)::PEtabODEProblemInfo
    model_size = _get_model_size(model.sys_mutated, model_info)
    gradient_method_use = _get_gradient_method(gradient_method, model_size,
                                               reuse_sensitivities)
    hessian_method_use = _get_hessian_method(hessian_method, model_size)
    FIM_method_use = _get_hessian_method(FIM_method, model_size)
    sensealg_use = _get_sensealg(sensealg, Val(gradient_method_use))
    sensealg_ss_use = _get_sensealg_ss(sensealg_ss, sensealg_use, model_info,
                                       Val(gradient_method_use))

    _check_method(gradient_method_use, :gradient)
    _check_method(hessian_method_use, :Hessian)
    _check_method(FIM_method_use, :FIM)

    split_use = _get_split_over_conditions(split_over_conditions, model_info)

    odesolver_use = _get_odesolver(odesolver, model_size, false, gradient_method_use,
                                   sensealg_use)
    odesolver_gradient_use = _get_odesolver(odesolver_gradient, model_size, true,
                                            gradient_method_use, sensealg_use;
                                            default_solver = odesolver_use)
    _ss_solver = _get_ss_solver(ss_solver)
    _ss_solver_gradient = _get_ss_solver(ss_solver_gradient)
    sparse_jacobian_use = _get_sparse_jacobian(sparse_jacobian, gradient_method_use,
                                               model_size)
    chunksize_use = isnothing(chunksize) ? 0 : chunksize

    # Several things to note here:
    # 1. p and u0 needs to be Float64 to avoid potential problems later, as sometimes
    #  they end up being Int due to SBML model file structure
    #  ODEFunction must be used because when going directly to ODEProblem MTKParameters
    #  are used as parameter struct, however, MTKParameters are not yet compatiable
    # 2. with SciMLSensitivity, and if remake is used to transform to parameter vector
    #  an error is thrown. The order of p is given by model_info.xindices.xids[:sys],
    #  (see conditions.jl for details) hence to set correct values for constant
    #  parameters the parameter map must be reorded.
    # 3. For ODEFunction an ODESystem is needed, hence ReactionSystems must be converted.
    btime = @elapsed begin
        odeproblem = _get_odeproblem(model.sys_mutated, model, model_info, specialize_level, sparse_jacobian_use)
        odeproblem_gradient = _get_odeproblem_gradient(odeproblem, gradient_method_use, sensealg_use)
    end
    _logging(:Build_ODEProblem, verbose; time = btime)

    # Cache to avoid allocations to as large degree as possible.
    cache = PEtabODEProblemCache(gradient_method_use, hessian_method_use, FIM_method_use,
                                 sensealg_use, model_info, model.ml_models, split_use, odeproblem)

    # To build the steady-state solvers the ODEProblem (specifically its Jacobian)
    # is needed (which is the same for odeproblem and odeproblem_gradient). Not yet comptiable with
    # UDE problems
    ss_solver_use = SteadyStateSolver(_ss_solver, odeproblem, odesolver_use)
    ss_solver_gradient_use = SteadyStateSolver(_ss_solver_gradient, odeproblem,
                                               odesolver_gradient_use)

    # For models with a neural net that feeds into model parameters, pre-build functions
    # for evaluating the neural-net and its Jacobian
    ml_models_pre_ode = _get_ml_models_pre_ode(model_info, cache)

    return PEtabODEProblemInfo(odeproblem, odeproblem_gradient, odesolver_use, odesolver_gradient_use,
                               ss_solver_use, ss_solver_gradient_use, gradient_method_use,
                               hessian_method_use, FIM_method_use, reuse_sensitivities,
                               sparse_jacobian_use, sensealg_use, sensealg_ss_use,
                               cache, split_use, chunksize_use, ml_models_pre_ode)
end

function _get_odeproblem(sys::ODEProblem, ::PEtabModel, model_info::ModelInfo,
                         specialize_level, ::Bool)::ODEProblem
    @unpack petab_parameters, petab_ml_parameters, xindices, model = model_info
    # Set constant parameter values (not done automatically as for a System based model)
    for (i, id) in pairs(petab_parameters.parameter_id)
        petab_parameters.estimate[i] == true && continue
        id in xindices.xids[:ml] && continue
        !haskey(sys.p, id) && continue
        sys.p[id] = petab_parameters.nominal_value[i]
    end
    odeproblem = remake(sys, u0 = sys.u0[:])
    # It matters that p follows the same order as in xids for correct indexing in the
    # adjoint gradient method
    odeproblem = remake(odeproblem, p = odeproblem.p[model_info.xindices.xids[:sys]])
    # Set potential constant neural net parameters in the ODE
    for ml_model_id in xindices.xids[:ml_in_ode]
        ml_model_id in xindices.xids[:ml_est] && continue
        set_ml_model_ps!((@view odeproblem.p[ml_model_id]), ml_model_id, model.ml_models[ml_model_id], model.paths)
    end
    return odeproblem
end
function _get_odeproblem(::ModelSystem, model::PEtabModel, model_info::ModelInfo, specialize_level, sparse_jacobian::Bool)::ODEProblem
    _set_const_parameters!(model, model_info.petab_parameters)
    @unpack sys_mutated, speciemap, parametermap, defined_in_julia = model
    _parametermap = _reorder_parametermap(parametermap, model_info.xindices.xids[:sys])
    _u0 = first.(speciemap) .=> 0.0
    odefun = ODEFunction(_to_odesystem(sys_mutated), first.(speciemap), first.(_parametermap); jac = true, sparse = sparse_jacobian)
    odeproblem = ODEProblem(odefun, last.(_u0), [0.0, 5e3], last.(_parametermap))
    return remake(odeproblem, p = Float64.(odeproblem.p), u0 = Float64.(odeproblem.u0))
end

function _to_odesystem(sys::ReactionSystem)::ODESystem
    return complete(convert(ODESystem, sys))
end
function _to_odesystem(sys::ODESystem)::ODESystem
    return sys
end

function _get_ml_models_pre_ode(model_info::ModelInfo, cache::PEtabODEProblemCache)::Dict{Symbol, Dict{Symbol, MLModelPreODE}}
    ml_models_pre_ode = Dict{Symbol, Dict{Symbol, MLModelPreODE}}()
    for (cid, maps_nn) in model_info.xindices.maps_ml_preode
        ml_model_pre_ode = Dict{Symbol, MLModelPreODE}()
        for (ml_model_id, map_ml_model) in maps_nn
            @unpack ninput_arguments, ninputs, noutputs = map_ml_model
            ml_model = model_info.model.ml_models[ml_model_id]
            # If parameters are constant, they only need to be assigned here, as when
            # building the cache xnn_not_est has the correct values.
            if ml_model_id in model_info.xindices.xids[:ml_est]
                pnn = cache.xnn[ml_model_id]
            else
                pnn = cache.xnn_constant[ml_model_id]
            end
            inputs = [DiffCache(zeros(Float64, n), levels = 2) for n in ninputs]
            outputs = DiffCache(zeros(Float64, noutputs), levels = 2)
            compute_forward! = let ml_model = ml_model, map_ml_model = map_ml_model, inputs = inputs, pnn = pnn
                (out, x) -> _net!(out, x, pnn, inputs, map_ml_model, ml_model)
            end

            # ReverseDiff.tape compatible (fastest on CPU, but only works if input is
            # known at compile-time). If there are multiple input arguments, they need
            # to be provided as a tuple, due to how ML models are imported
            if sum(map_ml_model.nxdynamic_inputs) == 0
                _inputs = (_get_input(map_ml_model, i) for i in eachindex(inputs))
                inputs_reverse = ninput_arguments == 1 ? first(_inputs) : Tuple(_inputs)
                compute_nn_rev! = let ml_model = ml_model, inputs_reverse = inputs_reverse
                    (out, x) -> _net_reversediff!(out, x, inputs_reverse, ml_model)
                end
                out = get_tmp(outputs, 1.0)
                _pnn = get_tmp(pnn, 1.0)
                tape = ReverseDiff.JacobianTape(compute_nn_rev!, out, _pnn)
            else
                tape = nothing
            end

            if ml_model_id in model_info.xindices.xids[:ml_est]
                nx = length(get_tmp(pnn, 1.0)) + sum(map_ml_model.nxdynamic_inputs)
            else
                nx = sum(map_ml_model.nxdynamic_inputs)
            end
            xarg = DiffCache(zeros(Float64, nx), levels = 2)
            jac_ml_model = zeros(Float64, noutputs, nx)
            ml_model_pre_ode[ml_model_id] = MLModelPreODE(compute_forward!, tape, jac_ml_model, outputs, inputs, xarg, [false])
        end
        ml_models_pre_ode[cid] = ml_model_pre_ode
    end
    return ml_models_pre_ode
end
