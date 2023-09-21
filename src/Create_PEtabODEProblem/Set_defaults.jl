#=
    Helper functions for setting default values when creating a PEtabODEProblem in case left unspecifed by the
    user.
=#


function set_ODESolver(ode_solver::Union{ODESolver, Nothing},
                       model_size::Symbol,
                       gradient_method::Symbol)::ODESolver

    if !isnothing(ode_solver)
        return ode_solver
    end

    if model_size === :Small
        return ODESolver(Rodas5P())
    end

    if model_size === :Medium
        return ODESolver(QNDF())
    end

    if model_size === :Large
        @warn "For large models we strongly recomend to compare different ODE-solvers instead of using default options"
        if gradient_method === :Adjoint || isnothing(gradient_method)
            return ODESolver(CVODE_BDF())
        else
            return ODESolver(KenCarp4())
        end
    end
end


function set_SteadyStateSolver(ss_options::Union{SteadyStateSolver, Nothing},
                               ode_solver::ODESolver)::SteadyStateSolver

    if !isnothing(ss_options)
        return ss_options
    end

     ss_solver = SteadyStateSolver(:Simulate,
                                   abstol=ode_solver.abstol / 100,
                                   reltol=ode_solver.reltol / 100)
    return ss_solver
end



function set_gradient_method(gradient_method::Union{Symbol, Nothing},
                           model_size::Symbol,
                           reuse_sensitivities::Bool)::Symbol

    if !isnothing(gradient_method)
        return gradient_method
    end

    if model_size === :Small
        return :ForwardDiff
    end

    if model_size === :Medium
        reuse_sensitivities == false && return :ForwardDiff
        reuse_sensitivities == true && return :ForwardEquations
    end

    if model_size === :Large
        if "SciMLSensitivity" ∉ string.(values(Base.loaded_modules)) 
            @warn "For large models adjoint sensitivity analysis is the best gradient method. To allow this to be set by default load SciMLSensitivity via using SciMLSensitivity"
            return :ForwardDiff
        end
        return :Adjoint
    end
end


function set_hessian_method(hessian_method::Union{Symbol, Nothing},
                          model_size::Symbol)

    if !isnothing(hessian_method)
        return hessian_method
    end

    if model_size === :Small
        return :ForwardDiff
    end

    if model_size === :Medium || model_size === :Large
        return :GaussNewton
    end
end


function set_sensealg(sensealg,
                     ::Val{:ForwardDiff})
    return nothing
end
function set_sensealg(sensealg,
                     ::Val{:ForwardEquations})
    if !isnothing(sensealg)
        @assert sensealg == :ForwardDiff "For gradient method :ForwardEquations allowed sensealg args are :ForwardDiff, ForwardSensitivity(), ForwardDiffSensitivity() not $sensealg"
        return sensealg
    end 
    return :ForwardDiff
end
