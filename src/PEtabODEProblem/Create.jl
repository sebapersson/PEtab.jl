function PEtabODEProblem(petab_model::PEtabModel;
                         odesolver::Union{Nothing, ODESolver} = nothing,
                         odesolver_gradient::Union{Nothing, ODESolver} = nothing,
                         ss_solver::Union{Nothing, SteadyStateSolver} = nothing,
                         ss_solver_gradient::Union{Nothing, SteadyStateSolver} = nothing,
                         gradient_method::Union{Nothing, Symbol} = nothing,
                         hessian_method::Union{Nothing, Symbol} = nothing,
                         FIM_method::Union{Nothing, Symbol} = nothing,
                         sparse_jacobian::Union{Nothing, Bool} = nothing,
                         specialize_level = SciMLBase.FullSpecialize,
                         sensealg = nothing,
                         sensealg_ss = nothing,
                         chunksize::Union{Nothing, Int64} = nothing,
                         split_over_conditions::Bool = false,
                         reuse_sensitivities::Bool = false,
                         verbose::Bool = true,
                         custom_values::Union{Nothing, Dict} = nothing)::PEtabODEProblem
    _logging(:Build_PEtabODEProblem, verbose; name = petab_model.modelname)

    # To bookeep parameters, measurements, etc...
    model_info = ModelInfo(petab_model, sensealg, custom_values)
    # All ODE-relevent info for the problem, e.g. solvers, gradient method ...
    probleminfo = PEtabODEProblemInfo(petab_model, model_info, odesolver, odesolver_gradient, ss_solver, ss_solver_gradient, gradient_method, hessian_method, FIM_method, sensealg, sensealg_ss, reuse_sensitivities, sparse_jacobian, specialize_level, chunksize, split_over_conditions, verbose)

    _logging(:Build_nllh, verbose)
    btime = @elapsed begin
        compute_cost = _get_nllh_f(probleminfo, model_info, false)
        compute_nllh = _get_nllh_f(probleminfo, model_info, false)
    end
    _logging(:Build_nllh, verbose; time = btime)

    _logging(:Build_gradient, verbose; method = probleminfo.gradient_method)
    btime = @elapsed begin
        method = probleminfo.gradient_method
        compute_grad!, compute_grad = get_grad_f(Val(method), probleminfo, model_info)
        compute_grad_nllh!, compute_grad_nllh = get_grad_f(Val(method), probleminfo, model_info)
        compute_nllh_grad = _get_nllh_grad_f(method, compute_grad, probleminfo, model_info)
    end
    _logging(:Build_gradient, verbose; time = btime)

    _logging(:Build_hessian, verbose; method = probleminfo.hessian_method)
    btime = @elapsed begin
        compute_hess!, compute_hess = _get_hess_f(probleminfo, model_info)
        compute_FIM!, compute_FIM = _get_hess_f(probleminfo, model_info; FIM = true)
    end
    _logging(:Build_hessian, verbose; time = btime)

    # TODO: Must refactor later
    compute_chi2 = (θ; as_array = false) -> begin
        _ = compute_cost(θ)
        if as_array == false
            return sum(measurement_info.chi2_values)
        else
            return measurement_info.chi2_values
        end
    end
    compute_residuals = (θ; as_array = false) -> begin
        _ = compute_cost(θ)
        if as_array == false
            return sum(measurement_info.residuals)
        else
            return measurement_info.residuals
        end
    end
    compute_simulated_values = (θ) -> begin
        _ = compute_cost(θ)
        return measurement_info.simulated_values
    end

    # Extract bounds and nominal parameter values
    @unpack θ_indices, parameter_info = model_info
    θ_names = θ_indices.xids[:estimate]
    lower_bounds = [parameter_info.lower_bounds[findfirst(x -> x == θ_names[i],
                                                          parameter_info.parameter_id)]
                    for i in eachindex(θ_names)]
    upper_bounds = [parameter_info.upper_bounds[findfirst(x -> x == θ_names[i],
                                                          parameter_info.parameter_id)]
                    for i in eachindex(θ_names)]
    θ_nominal = [parameter_info.nominal_value[findfirst(x -> x == θ_names[i],
                                                        parameter_info.parameter_id)]
                 for i in eachindex(θ_names)]
    transformθ!(lower_bounds, θ_names, θ_indices, reverse_transform = true)
    transformθ!(upper_bounds, θ_names, θ_indices, reverse_transform = true)
    θ_nominalT = transformθ(θ_nominal, θ_names, θ_indices, reverse_transform = true)

    petab_problem = PEtabODEProblem(compute_cost,
                                    compute_nllh,
                                    compute_chi2,
                                    compute_grad!,
                                    compute_grad,
                                    compute_grad_nllh!,
                                    compute_grad_nllh,
                                    compute_hess!,
                                    compute_hess,
                                    compute_FIM!,
                                    compute_FIM,
                                    compute_nllh_grad,
                                    compute_simulated_values,
                                    compute_residuals,
                                    :Only,
                                    probleminfo.gradient_method,
                                    probleminfo.hessian_method,
                                    probleminfo.FIM_method,
                                    Int64(length(θ_names)),
                                    θ_names,
                                    θ_nominal,
                                    θ_nominalT,
                                    lower_bounds,
                                    upper_bounds,
                                    petab_model,
                                    probleminfo.solver,
                                    probleminfo.solver_gradient,
                                    probleminfo.ss_solver,
                                    probleminfo.ss_solver_gradient,
                                    model_info.θ_indices,
                                    model_info.simulation_info,
                                    probleminfo.odeproblem,
                                    probleminfo.split_over_conditions,
                                    model_info.prior_info,
                                    model_info.parameter_info,
                                    probleminfo.petab_ODE_cache,
                                    model_info.measurement_info,
                                    probleminfo.petab_ODESolver_cache)
    return petab_problem
end

function _get_nllh_f(probleminfo::PEtabODEProblemInfo, model_info::ModelInfo, compute_residuals::Bool)::Function
    _compute_nllh = let pinfo = probleminfo, minfo = model_info, compute_residuals = compute_residuals
        (x) -> compute_cost(x, pinfo, minfo, [:all], true, false, compute_residuals)
    end
    return _compute_nllh
end

function get_grad_f(method, probleminfo::PEtabODEProblemInfo, model_info::ModelInfo)::Tuple{Function, Function}
    @unpack split_over_conditions, gradient_method, chunksize = probleminfo
    @unpack sensealg, petab_ODE_cache = probleminfo
    @unpack θ_dynamic = petab_ODE_cache

    if gradient_method == :ForwardDiff
        _nllh_not_solve = _get_nllh_not_solve(probleminfo, model_info; compute_gradient_not_solve_autodiff = true)

        if split_over_conditions == false
            _nllh_solve = let pinfo = probleminfo, minfo = model_info
                @unpack θ_sd, θ_observable, θ_non_dynamic = pinfo.petab_ODE_cache
                (x) -> compute_cost_solve_ODE(x, θ_sd, θ_observable, θ_non_dynamic, pinfo, minfo; compute_gradient_θ_dynamic = true, exp_id_solve = [:all])
            end

            chunksize_use = _get_chunksize(chunksize, θ_dynamic)
            cfg = ForwardDiff.GradientConfig(_nllh_solve, θ_dynamic, chunksize_use)
            _compute_gradient! = let _nllh_not_solve = _nllh_not_solve, _nllh_solve = _nllh_solve, cfg = cfg, minfo = model_info, pinfo = probleminfo
                (grad, x; isremade = false) -> compute_gradient_autodiff!(grad, x, _nllh_not_solve, _nllh_solve, cfg, minfo, pinfo; isremade = isremade)
            end
        end

        if split_over_conditions == true
            _nllh_solve = let pinfo = probleminfo, minfo = model_info
                @unpack θ_sd, θ_observable, θ_non_dynamic = pinfo.petab_ODE_cache
                (x, eid) -> compute_cost_solve_ODE(x, θ_sd, θ_observable, θ_non_dynamic, pinfo, minfo; compute_gradient_θ_dynamic = true, exp_id_solve = eid)
            end
            _compute_gradient! = let _nllh_not_solve = _nllh_not_solve, _nllh_solve = _nllh_solve, minfo = model_info, pinfo = probleminfo
                (g, θ) -> compute_gradient_autodiff_split!(g, θ, _nllh_not_solve, _nllh_solve, pinfo, minfo)
            end
        end
    end

    if gradient_method === :ForwardEquations
        chunksize_use = _get_chunksize(chunksize, θ_dynamic)
        if sensealg == :ForwardDiff && split_over_conditions == false
            _solve_conditions! = let pinfo = probleminfo, minfo = model_info
                (sols, x) -> solve_ode_all_conditions!(sols, x, pinfo, minfo; save_at_observed_t = true, exp_id_solve = [:all], compute_forward_sensitivites_ad = true)
            end
            cfg = ForwardDiff.JacobianConfig(_solve_conditions!, petab_ODE_cache.sol_values, θ_dynamic, chunksize_use)
        end
        if sensealg == :ForwardDiff && split_over_conditions == true
            _solve_conditions! = let pinfo = probleminfo, minfo = model_info
                (sols, x, eid) -> solve_ode_all_conditions!(sols, x, pinfo, minfo; save_at_observed_t = true, exp_id_solve = eid, compute_forward_sensitivites_ad = true)
            end
            cfg = ForwardDiff.JacobianConfig(_solve_conditions!, petab_ODE_cache.sol_values, petab_ODE_cache.θ_dynamic, chunksize_use)
        end
        if sensealg != :ForwardDiff
            _solve_conditions! = let pinfo = probleminfo, minfo = model_info
                (x, eid) -> solve_ode_all_conditions!(minfo, x, pinfo; save_at_observed_t = true, exp_id_solve = eid, compute_forward_sensitivites = true)
            end
            cfg = nothing
        end

        _nllh_not_solve = _get_nllh_not_solve(probleminfo, model_info; compute_gradient_not_solve_forward = true)

        _compute_gradient! = let _nllh_not_solve = _nllh_not_solve, _solve_conditions! = _solve_conditions!, minfo = model_info, pinfo = probleminfo, cfg = cfg
            (g, θ; isremade = false) -> compute_gradient_forward_equations!(g, θ, _nllh_not_solve, _solve_conditions!, pinfo, minfo, cfg; exp_id_solve = [:all], isremade = isremade)
        end
    end

    _compute_gradient = let _compute_gradient! = _compute_gradient!
        (x) -> begin
            gradient = zeros(Float64, length(x))
            _compute_gradient!(gradient, x)
            return gradient
        end
    end
    return _compute_gradient!, _compute_gradient
end

function _get_hess_f(probleminfo::PEtabODEProblemInfo, model_info::ModelInfo; return_jacobian::Bool = false, FIM::Bool = false)::Tuple{Function, Function}
    @unpack hessian_method, split_over_conditions, chunksize, petab_ODE_cache = probleminfo
    @unpack θ_dynamic = petab_ODE_cache
    if FIM == true
        hessian_method = probleminfo.FIM_method
    end

    if hessian_method === :ForwardDiff
        if split_over_conditions == false
            _nllh = let pinfo = probleminfo, minfo = model_info
                (x) -> compute_cost(x, pinfo, minfo, [:all], false, true, false)
            end

            nestimate = length(model_info.θ_indices.xids[:estimate])
            chunksize_use = _get_chunksize(chunksize, zeros(nestimate))
            cfg = ForwardDiff.HessianConfig(_nllh, zeros(nestimate), chunksize_use)
            _compute_hessian! = let _nllh = _nllh, cfg = cfg, minfo = model_info
                (H, x) -> compute_hessian!(H, x, _nllh, cfg, minfo)
            end
        end

        if split_over_conditions == true
            _nllh = let pinfo = probleminfo, minfo = model_info
                (x, eid) -> compute_cost(x, pinfo, minfo, eid, false, true, false)
            end

            _compute_hessian! = let _nllh = _nllh, minfo = model_info
                (H, x) -> compute_hessian_split!(H, x, _nllh, minfo)
            end
        end
    end

    if hessian_method === :BlockForwardDiff
        _nllh_not_solve = _get_nllh_not_solve(probleminfo, model_info; compute_gradient_not_solve_autodiff = true)

        if split_over_conditions == false
            _nllh_solve = let pinfo = probleminfo, minfo = model_info
                @unpack θ_sd, θ_observable, θ_non_dynamic = pinfo.petab_ODE_cache
                (x) -> compute_cost_solve_ODE(x, θ_sd, θ_observable, θ_non_dynamic, pinfo, minfo; compute_gradient_θ_dynamic = true, exp_id_solve = [:all])
            end

            chunksize_use = _get_chunksize(chunksize, θ_dynamic)
            cfg = ForwardDiff.HessianConfig(_nllh_solve, θ_dynamic, chunksize_use)
            _compute_hessian! = let _nllh_solve = _nllh_solve, _nllh_not_solve = _nllh_not_solve, pinfo = probleminfo, minfo = model_info, cfg = cfg
                (H, x) -> compute_hessian_block!(H, x, _nllh_not_solve, _nllh_solve, pinfo, minfo, cfg; exp_id_solve = [:all])
            end
        end

        if split_over_conditions == true
            _nllh_solve = let pinfo = probleminfo, minfo = model_info
                @unpack θ_sd, θ_observable, θ_non_dynamic = pinfo.petab_ODE_cache
                (x, eid) -> compute_cost_solve_ODE(x, θ_sd, θ_observable, θ_non_dynamic, pinfo, minfo,  compute_gradient_θ_dynamic = true, exp_id_solve = eid)
            end

            _compute_hessian! = let _nllh_solve = _nllh_solve, _nllh_not_solve = _nllh_not_solve, pinfo = probleminfo, minfo = model_info
                (H, x) -> compute_hessian_block_split!(H, x, _nllh_not_solve, _nllh_solve, pinfo, minfo; exp_id_solve = [:all])
            end
        end
    end

    if hessian_method == :GaussNewton
        if split_over_conditions == false
            _solve_conditions! = let pinfo = probleminfo, minfo = model_info
                (sols, x) -> solve_ode_all_conditions!(sols, x, pinfo, minfo; save_at_observed_t = true, exp_id_solve = [:all], compute_forward_sensitivites_ad = true)
            end
        end
        if split_over_conditions == true
            _solve_conditions! = let pinfo = probleminfo, minfo = model_info
                (sols, x, eid) -> solve_ode_all_conditions!(sols, x, pinfo, minfo; save_at_observed_t = true, exp_id_solve = eid, compute_forward_sensitivites_ad = true)
            end
        end
        chunksize_use = _get_chunksize(chunksize, θ_dynamic)
        cfg = cfg = ForwardDiff.JacobianConfig(_solve_conditions!, petab_ODE_cache.sol_values, petab_ODE_cache.θ_dynamic, chunksize_use)

        _residuals_not_solve! = let pinfo = probleminfo, minfo = model_info
            iθ_sd, iθ_observable, iθ_non_dynamic, _ = get_index_parameters_not_ODE(model_info.θ_indices)
            (residuals, x) -> begin
                compute_residuals_not_solve_ode!(residuals, x[iθ_sd], x[iθ_observable], x[iθ_non_dynamic], pinfo, minfo; exp_id_solve = [:all])
            end
        end

        xnot_ode = zeros(Float64, length(model_info.θ_indices.xids[:not_system]))
        cfg_notsolve = ForwardDiff.JacobianConfig(_residuals_not_solve!, petab_ODE_cache.residuals_gn, xnot_ode, ForwardDiff.Chunk(xnot_ode))
        _compute_hessian! = let _residuals_not_solve! = _residuals_not_solve!, pinfo = probleminfo, minfo = model_info, cfg = cfg, cfg_notsolve = cfg_notsolve, return_jacobian = return_jacobian, _solve_conditions! = _solve_conditions!
            (H, x; isremade = false) -> compute_GaussNewton_hessian!(H, x, _residuals_not_solve!, _solve_conditions!, pinfo, minfo, cfg, cfg_notsolve; exp_id_solve = [:all], isremade = isremade, return_jacobian = return_jacobian)
        end
    end

    _compute_hessian = (x) -> begin
        hessian = zeros(Float64, length(x), length(x))
        _compute_hessian!(hessian, x)
        return hessian
    end
    return _compute_hessian!, _compute_hessian
end

function _get_nllh_grad_f(gradient_method::Symbol, compute_grad::Function, probleminfo::PEtabODEProblemInfo, model_info::ModelInfo)::Function
    compute_gradient_not_solve_autodiff = gradient_method == :ForwardDiff
    compute_gradient_not_solve_forward = gradient_method == :ForwardEquations
    compute_gradient_not_solve_adjoint = gradient_method == :Adjoint

    _nllh_not_solve = _get_nllh_not_solve(probleminfo, model_info; compute_gradient_not_solve_autodiff = compute_gradient_not_solve_autodiff, compute_gradient_not_solve_forward = compute_gradient_not_solve_forward, compute_gradient_not_solve_adjoint = compute_gradient_not_solve_adjoint)
    _compute_nllh_grad = (x) -> begin
        grad = compute_grad(x)
        nllh = _nllh_not_solve(x)
        return nllh, grad
    end
    return _compute_nllh_grad
end

function _get_nllh_not_solve(probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                             compute_gradient_not_solve_autodiff::Bool = false,
                             compute_gradient_not_solve_forward::Bool = false,
                             compute_gradient_not_solve_adjoint::Bool = false)::Function
    _nllh_not_solve = let pinfo = probleminfo, minfo = model_info
        # TODO: Precompute!
        iθ_sd, iθ_observable, iθ_non_dynamic, _ = get_index_parameters_not_ODE(minfo.θ_indices)
        (x) -> compute_cost_not_solve_ODE(x[iθ_sd], x[iθ_observable], x[iθ_non_dynamic], pinfo, minfo; exp_id_solve = [:all], compute_gradient_not_solve_autodiff = compute_gradient_not_solve_autodiff, compute_gradient_not_solve_forward = compute_gradient_not_solve_forward, compute_gradient_not_solve_adjoint = compute_gradient_not_solve_adjoint)
    end
    return _nllh_not_solve
end

function PEtabODEProblemInfo(petab_model::PEtabModel, model_info::ModelInfo, odesolver, odesolver_gradient,  ss_solver, ss_solver_gradient, gradient_method, hessian_method, FIM_method, sensealg, sensealg_ss, reuse_sensitivities::Bool, sparse_jacobian, specialize_level, chunksize, split_over_conditions::Bool, verbose::Bool)::PEtabODEProblemInfo
    model_size = _get_model_size(petab_model.sys_mutated, model_info)
    gradient_method_use = _get_gradient_method(gradient_method, model_size, reuse_sensitivities)
    hessian_method_use = _get_hessian_method(hessian_method, model_size)
    FIM_method_use = _get_hessian_method(FIM_method, model_size)
    sensealg_use = _get_sensealg(sensealg, Val(gradient_method_use))
    sensealg_ss_use = _get_sensealg_ss(sensealg_ss, sensealg_use, model_info, Val(gradient_method_use))

    _check_method(gradient_method_use, :gradient)
    _check_method(hessian_method_use, :Hessian)
    _check_method(FIM_method_use, :FIM)

    odesolver_use = _get_odesolver(odesolver, model_size, gradient_method_use)
    odesolver_gradient_use = _get_odesolver(odesolver_gradient, model_size, gradient_method_use; default_solver = odesolver_use)
    _ss_solver = _get_ss_solver(ss_solver, odesolver_use)
    _ss_solver_gradient = _get_ss_solver(ss_solver_gradient, odesolver_gradient_use)
    sparse_jacobian_use = _get_sparse_jacobian(sparse_jacobian, model_size)
    chunksize_use = isnothing(chunksize) ? 0 : chunksize

    # Cache to avoid allocations to as large degree as possible. TODO: Refactor into single
    petab_ODE_cache = PEtabODEProblemCache(gradient_method_use, hessian_method_use, FIM_method_use, sensealg_use, model_info)
    petab_ODESolver_cache = PEtabODESolverCache(gradient_method_use, hessian_method_use, model_info)

    _logging(:Build_ODEProblem, verbose)
    btime = @elapsed begin
        _set_constant_ode_parameters!(petab_model, model_info.parameter_info)
        @unpack sys_mutated, statemap, parametermap, defined_in_julia = petab_model
        if sys_mutated isa ODESystem && defined_in_julia == false
            SL = specialize_level
            _oprob = ODEProblem{true, SL}(sys_mutated, statemap, [0.0, 5e3], parametermap; jac = true, sparse = sparse_jacobian_use)
        else
            # For ReactionSystem there is bug if I try to set specialize_level. Also,
            # statemap must somehow be a vector. TODO: Test with MTKv9
            u0map_tmp = zeros(Float64, length(petab_model.statemap))
            _oprob = ODEProblem(sys_mutated, u0map_tmp, [0.0, 5e3], parametermap; jac = true, sparse = sparse_jacobian_use)
        end
        # Ensure correct types for further computations
        oprob = remake(_oprob, p = Float64.(_oprob.p), u0 = Float64.(_oprob.u0))
        oprob_gradient = _get_odeproblem_gradient(oprob, gradient_method_use, sensealg_use)
    end
    _logging(:Build_ODEProblem, verbose; time = btime)

    # To build the steady-state solvers the ODEProblem (specifically its Jacobian)
    # is needed (which is the same for oprob and oprob_gradient)
    ss_solver_use = _get_steady_state_solver(_ss_solver, oprob)
    ss_solver_gradient_use = _get_steady_state_solver(_ss_solver_gradient, oprob)

    return PEtabODEProblemInfo(oprob, oprob_gradient, odesolver_use, odesolver_gradient_use, ss_solver_use, ss_solver_gradient_use, gradient_method_use, hessian_method_use, FIM_method_use, reuse_sensitivities, sparse_jacobian_use, sensealg_use, sensealg_ss_use, petab_ODE_cache, petab_ODESolver_cache, split_over_conditions, chunksize_use)
end
