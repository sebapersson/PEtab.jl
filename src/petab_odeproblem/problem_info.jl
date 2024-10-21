const GRADIENT_METHODS = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint]
const HESSIAN_METHODS = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
const FIM_METHODS = [nothing, :ForwardDiff, :GaussNewton]

function PEtabODEProblemInfo(model::PEtabModel, model_info::ModelInfo, odesolver,
                             odesolver_gradient, ss_solver, ss_solver_gradient,
                             gradient_method, hessian_method, FIM_method, sensealg,
                             sensealg_ss, reuse_sensitivities::Bool, sparse_jacobian,
                             specialize_level, chunksize, split_over_conditions,
                             verbose::Bool)::PEtabODEProblemInfo
    _logging(:Build_ODEProblem, verbose)
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

    odesolver_use = _get_odesolver(odesolver, model_size, gradient_method_use)
    odesolver_gradient_use = _get_odesolver(odesolver_gradient, model_size,
                                            gradient_method_use;
                                            default_solver = odesolver_use)
    _ss_solver = _get_ss_solver(ss_solver)
    _ss_solver_gradient = _get_ss_solver(ss_solver_gradient)
    sparse_jacobian_use = _get_sparse_jacobian(sparse_jacobian, gradient_method_use,
                                               model_size)
    chunksize_use = isnothing(chunksize) ? 0 : chunksize

    _logging(:Build_ODEProblem, verbose)
    btime = @elapsed begin
        oprob = _get_odeproblem(model.sys_mutated, model, model_info, specialize_level,
                                sparse_jacobian_use)
        oprob_gradient = _get_odeproblem_gradient(oprob, gradient_method_use, sensealg_use)
    end
    _logging(:Build_ODEProblem, verbose; time = btime)

    # Cache to avoid allocations to as large degree as possible.
    cache = PEtabODEProblemCache(gradient_method_use, hessian_method_use, FIM_method_use,
                                 sensealg_use, model_info, model.nn, oprob)

    # To build the steady-state solvers the ODEProblem (specifically its Jacobian)
    # is needed (which is the same for oprob and oprob_gradient). Not yet comptiable with
    # UDE problems
    ss_solver_use = SteadyStateSolver(_ss_solver, oprob, odesolver_use)
    ss_solver_gradient_use = SteadyStateSolver(_ss_solver_gradient, oprob,
                                               odesolver_gradient_use)

    return PEtabODEProblemInfo(oprob, oprob_gradient, odesolver_use, odesolver_gradient_use,
                               ss_solver_use, ss_solver_gradient_use, gradient_method_use,
                               hessian_method_use, FIM_method_use, reuse_sensitivities,
                               sparse_jacobian_use, sensealg_use, sensealg_ss_use,
                               cache, split_use, chunksize_use)
end

function _get_odeproblem(sys::ODEProblem, ::PEtabModel, model_info::ModelInfo,
                         specialize_level, ::Bool)::ODEProblem
    for (i, id) in pairs(model_info.petab_parameters.parameter_id)
        model_info.petab_parameters.estimate[i] == true && continue
        !haskey(sys.p, id) && continue
        sys.p[id] = model_info.petab_parameters.nominal_value[i]
    end
    _sys = remake(sys, u0 = sys.u0[:])
    return _sys
end
function _get_odeproblem(sys, model::PEtabModel, model_info::ModelInfo, specialize_level,
                         sparse_jacobian::Bool)::ODEProblem
    _set_const_parameters!(model, model_info.petab_parameters)
    @unpack speciemap, parametermap, defined_in_julia = model
    if sys isa ODESystem && defined_in_julia == false
        SL = specialize_level
        _oprob = ODEProblem{true, SL}(sys, speciemap, [0.0, 5e3], parametermap;
                                      jac = true, sparse = sparse_jacobian)
    else
        # For ReactionSystem there is bug if I try to set specialize_level. Also,
        # speciemap must somehow be a vector. TODO: Test with MTKv9
        u0map_tmp = zeros(Float64, length(model.speciemap))
        _oprob = ODEProblem(sys, u0map_tmp, [0.0, 5e3], parametermap;
                            jac = true, sparse = sparse_jacobian)
    end
    # Ensure correct types for further computations. Long-term we plan to here
    # transition to the SciMLStructures interface, but that has to wait for
    # SciMLSensitivity
    if _oprob.p isa ModelingToolkit.MTKParameters
        _p = _oprob.p.tunable .|> Float64
        oprob = remake(_oprob, p = _p, u0 = Float64.(_oprob.u0))
    else
        oprob = remake(_oprob, p = Float64.(_oprob.p), u0 = Float64.(_oprob.u0))
    end
    return oprob
end
