function _check_method(method::Symbol, whatcheck::Symbol)::Nothing
    if whatcheck == :gradient
        allowed_methods = GRADIENT_METHODS
        if method == :Adjoint
            @assert "SciMLSensitivity" ∈ string.(values(Base.loaded_modules)) "To use "*
            "adjoint sensitivity analysis SciMLSensitivity must be loaded"
        end
    elseif whatcheck == :Hessian
        allowed_methods = HESSIAN_METHODS
    elseif whatcheck == :FIM
        allowed_methods = FIM_METHODS
    end

    if !(method in allowed_methods)
        throw(PEtabInputError("$(method) is an allowed $(whatcheck) option. Allowed " *
                              "options are $(allowed_methods)"))
    end
end

function _get_model_size(sys::ModelSystem,
                         model_info::ModelInfo)::Symbol
    nODEs = length(unknowns(sys))
    # When looking at number of dynamic parameters it is important to look at how
    # many dynamic parmeters are estimated per condition. For example, xdynamic might
    # have a size of 80, but only 10 parameters are estimated per condition. In these
    # cases the model is not large in the parameter sense, and adjoint should not be
    # used, rather ForwardDiff with split_over_conditions = true
    n_xdynamic_sys = _get_n_xdynamic_sys(model_info)
    if nODEs ≤ 15 && n_xdynamic_sys ≤ 20
        return :Small
    elseif nODEs ≤ 50 && n_xdynamic_sys ≤ 70
        return :Medium
    else
        return :Large
    end
end

function _get_split_over_conditions(split::Union{Nothing, Bool},
                                    model_info::ModelInfo)::Bool
    !isnothing(split) && return split
    nxdynamic_sys = _get_n_xdynamic_sys(model_info)
    nxdynamic = length(model_info.xindices.xids[:dynamic])
    if nxdynamic ≥ 2 * nxdynamic_sys
        return true
    else
        return false
    end
end

function _get_n_xdynamic_sys(model_info::ModelInfo)::Int64
    n_constant = 0
    n_xdynamic_sys = 0
    for (i, condition_map) in pairs(collect(values(model_info.xindices.condition_maps)))
        if i == 1
            n_constant = length(condition_map.isys_all_conditions)
        end
        if (n_constant + length(condition_map.ix_condition)) > n_xdynamic_sys
            n_xdynamic_sys = n_constant + length(condition_map.ix_condition)
        end
    end
    return n_xdynamic_sys
end

function _get_gradient_method(method::Union{Symbol, Nothing}, model_size::Symbol,
                              reuse_sensitivities::Bool)::Symbol
    !isnothing(method) && return method
    if model_size == :Small
        return :ForwardDiff
    end
    if model_size == :Medium
        reuse_sensitivities == false && return :ForwardDiff
        reuse_sensitivities == true && return :ForwardEquations
    end
    if model_size === :Large
        if !("SciMLSensitivity" in string.(values(Base.loaded_modules)))
            @warn "For large models adjoint sensitivity analysis is the best gradient " *
                  "method. To use this method load SciMLSensitivity"
            return :ForwardDiff
        end
        return :Adjoint
    end
end

function _get_hessian_method(method::Union{Symbol, Nothing}, model_size::Symbol)::Symbol
    !isnothing(method) && return method
    if model_size == :Small
        return :ForwardDiff
    end
    if model_size == :Medium || model_size == :Large
        return :GaussNewton
    end
end

function _get_odesolver(solver::Union{ODESolver, Nothing}, model_size::Symbol,
                        gradient::Bool, gradient_method::Symbol, sensealg;
                        default_solver = nothing)::ODESolver
    !isnothing(solver) && gradient == false && return solver
    solver = isnothing(solver) ? default_solver : solver
    # Only pure Julia solvers are compatible with autodiff (ForwardDiff)
    autodiff = (gradient_method == :ForwardDiff) ||
               (gradient_method == :ForwardEquations && sensealg == :ForwardDiff)
    if gradient && !isnothing(solver) && !SciMLBase.isautodifferentiable(solver.solver) &&
       autodiff
        throw(PEtab.PEtabInputError("$solver is not compatible with automatic \
            differentiation. Either use a ForwardDiff compatible solver, e.g. most Julia \
            solvers like QNDF and Rodas5P, or a non-autodiff gradient method like :Adjoint \
            or :ForwardEquations with sensealg = SciMLSensitivity.ForwardSensitivity()"))
    elseif !isnothing(solver)
        return solver
    end
    if model_size == :Small
        return ODESolver(Rodas5P())
    end
    if model_size == :Medium
        return ODESolver(QNDF())
    end
    if model_size == :Large
        @warn "For large models we strongly recomend to compare different ODE-solvers " *
              "instead of using default options."
        # When not setting gradient solver
        if gradient_method == :Adjoint
            return ODESolver(CVODE_BDF())
        else
            return ODESolver(KenCarp4())
        end
    end
end

function _get_ss_solver(ss_solver::Union{SteadyStateSolver, Nothing})::SteadyStateSolver
    !isnothing(ss_solver) && return ss_solver
    return SteadyStateSolver(:Simulate)
end

function _get_sparse_jacobian(sparse::Union{Bool, Nothing}, gradient_method::Symbol,
                              model_size::Symbol)::Bool
    !isnothing(sparse) && return sparse
    gradient_method in [:ForwardDiff, :ForwardEquations] && return false
    model_size == :Large && return true
    return false
end

_get_sensealg(sensealg, ::Val{:ForwardDiff})::Nothing = nothing
function _get_sensealg(sensealg, ::Val{:ForwardEquations})::Symbol
    allowed_methods = [":ForwardDiff", "ForwardSensitivity()", "ForwardDiffSensitivity()"]
    if !isnothing(sensealg) && sensealg != :ForwardDiff
        throw(PEtabInputError("For gradient method :ForwardEquations allowed sensealg" *
                              "arguments are $(allowed_methods). To use the latter two " *
                              "methods SciMLSensitivity must be loaded."))
    end
    return :ForwardDiff
end

function _get_sensealg_ss(sensealg_ss, sensealg, ::ModelInfo, gradient_method)::Nothing
    return nothing
end

function _get_odeproblem_gradient(odeproblem::ODEProblem, gradient_method::Symbol,
                                  sensealg)::ODEProblem
    # This is only relevant when sensealg is from SciMLSensitivity, but the code cannot
    # reach this point with sensealg ForwardSensitivity() or ForwardDiffSensitivity()
    # with SciMLSensitivity loaded, and if SciMLSensitivity is not loaded _get_sensealg
    # will throw an error
    return odeproblem
end

function _get_chunksize(chunksize::Int64, xdynamic::Vector{<:AbstractFloat})
    if chunksize == 0
        return ForwardDiff.Chunk(xdynamic)
    else
        return ForwardDiff.Chunk(chunksize)
    end
end
