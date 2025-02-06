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
                                 sensealg_use, model_info, model.nnmodels, split_use, oprob)

    # To build the steady-state solvers the ODEProblem (specifically its Jacobian)
    # is needed (which is the same for oprob and oprob_gradient). Not yet comptiable with
    # UDE problems
    ss_solver_use = SteadyStateSolver(_ss_solver, oprob, odesolver_use)
    ss_solver_gradient_use = SteadyStateSolver(_ss_solver_gradient, oprob,
                                               odesolver_gradient_use)

    # For models with a neural net that feeds into model parameters, pre-build functions
    # for evaluating the neural-net and its Jacobian
    f_nns_preode = _get_f_nns_preode(model_info, cache)

    return PEtabODEProblemInfo(oprob, oprob_gradient, odesolver_use, odesolver_gradient_use,
                               ss_solver_use, ss_solver_gradient_use, gradient_method_use,
                               hessian_method_use, FIM_method_use, reuse_sensitivities,
                               sparse_jacobian_use, sensealg_use, sensealg_ss_use,
                               cache, split_use, chunksize_use, f_nns_preode)
end

function _get_odeproblem(sys::ODEProblem, ::PEtabModel, model_info::ModelInfo,
                         specialize_level, ::Bool)::ODEProblem
    @unpack petab_parameters, xindices, model = model_info
    for (i, id) in pairs(petab_parameters.parameter_id)
        petab_parameters.estimate[i] == true && continue
        id in xindices.xids[:nn] && continue
        !haskey(sys.p, id) && continue
        sys.p[id] = petab_parameters.nominal_value[i]
    end
    _sys = remake(sys, u0 = sys.u0[:])
    # It matters that p follows the same order as in xids for correct indexing in the
    # adjoint gradient method
    __sys = remake(_sys, p = _sys.p[model_info.xindices.xids[:sys]])
    # Set potential constant neural net parameters in the ODE
    for netid in xindices.xids[:nn_in_ode]
        netid in xindices.xids[:nn_est] && continue
        psfile_path = joinpath(model.paths[:dirmodel], petab_parameters.nn_parameters_files[netid])
        set_ps_net!((@view __sys.p[netid]), psfile_path, model.nnmodels[netid].nn)
    end
    return __sys
end
function _get_odeproblem(sys, model::PEtabModel, model_info::ModelInfo, specialize_level,
                         sparse_jacobian::Bool)::ODEProblem
    _set_const_parameters!(model, model_info.petab_parameters)
    @unpack speciemap, parametermap, defined_in_julia = model
    if sys isa ODESystem && defined_in_julia == false
        # With MTK v9 speciemap must somehow be a vector.
        SL = specialize_level
        u0map_tmp = zeros(Float64, length(model.speciemap))
        _oprob = ODEProblem{true, SL}(sys, u0map_tmp, [0.0, 5e3], parametermap;
                                      jac = true, sparse = sparse_jacobian)
    else
        # For ReactionSystem and there is bug if I try to set specialize_level.
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

function _get_f_nns_preode(model_info::ModelInfo, cache::PEtabODEProblemCache)::Dict{Symbol, Dict{Symbol, NNPreODE}}
    f_nns_preode = Dict{Symbol, Dict{Symbol, NNPreODE}}()
    for (cid, maps_nn) in model_info.xindices.maps_nn_preode
        f_nn_preode = Dict{Symbol, NNPreODE}()
        for (netid, map_nn) in maps_nn
            @unpack ninputs, noutputs = map_nn
            nnmodel = model_info.model.nnmodels[netid]
            # If parameters are constant, they only need to be assigned here, as when
            # building the cache xnn_not_est has the correct values.
            if netid in model_info.xindices.xids[:nn_est]
                pnn = cache.xnn[netid]
            else
                pnn = cache.xnn_constant[netid]
            end
            inputs = DiffCache(zeros(Float64, ninputs), levels = 2)
            outputs = DiffCache(zeros(Float64, noutputs), levels = 2)
            compute_nn! = let nnmodel = nnmodel, map_nn = map_nn, inputs = inputs, pnn = pnn
                (out, x) -> _net!(out, x, pnn, inputs, map_nn, nnmodel)
            end

            # ReverseDiff.tape compatible (fastest on CPU, but only works if input is
            # known at compile-time)
            if map_nn.nxdynamic_inputs == 0
                if map_nn.file_input == false
                    inputs_rev = map_nn.constant_inputs[map_nn.iconstant_inputs]
                else
                    inputs_rev = map_nn.constant_inputs
                end
                compute_nn_rev! = let nnmodel = nnmodel, inputs_rev = inputs_rev
                    (out, x) -> _net_reversediff!(out, x, inputs_rev, nnmodel)
                end
                out = get_tmp(outputs, 1.0)
                _pnn = get_tmp(pnn, 1.0)
                tape = ReverseDiff.JacobianTape(compute_nn_rev!, out, _pnn)
            else
                tape = nothing
            end

            if netid in model_info.xindices.xids[:nn_est]
                nx = length(get_tmp(pnn, 1.0)) + map_nn.nxdynamic_inputs
            else
                nx = map_nn.nxdynamic_inputs
            end
            xarg = DiffCache(zeros(Float64, nx), levels = 2)
            jac_nn = zeros(Float64, noutputs, nx)
            f_nn_preode[netid] = NNPreODE(compute_nn!, tape, jac_nn, outputs, inputs, xarg, [false])
        end
        f_nns_preode[cid] = f_nn_preode
    end
    return f_nns_preode
end
