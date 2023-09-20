#=
    Helper functions for setting default values when creating a PEtabODEProblem in case left unspecifed by the
    user.
=#


function setODESolver(ode_solver::Union{ODESolver, Nothing},
                             modelSize::Symbol,
                             gradient_method::Symbol)::ODESolver

    if !isnothing(ode_solver)
        return ode_solver
    end

    if modelSize === :Small
        return ODESolver(Rodas5P())
    end

    if modelSize === :Medium
        return ODESolver(QNDF())
    end

    if modelSize === :Large
        @warn "For large models we strongly recomend to compare different ODE-solvers instead of using default options"
        if gradient_method === :Adjoint || isnothing(gradient_method)
            return ODESolver(CVODE_BDF())
        else
            return ODESolver(KenCarp4())
        end
    end
end


function setSteadyStateSolver(ssOptions::Union{SteadyStateSolver, Nothing},
                                     ode_solver::ODESolver)::SteadyStateSolver

    if !isnothing(ssOptions)
        return ssOptions
    end

     ss_solver = SteadyStateSolver(:Simulate,
                                                abstol=ode_solver.abstol / 100,
                                                reltol=ode_solver.reltol / 100)
    return ss_solver
end



function setGradientMethod(gradient_method::Union{Symbol, Nothing},
                           modelSize::Symbol,
                           reuse_sensitivities::Bool)::Symbol

    if !isnothing(gradient_method)
        return gradient_method
    end

    if modelSize === :Small
        return :ForwardDiff
    end

    if modelSize === :Medium
        reuse_sensitivities == false && return :ForwardDiff
        reuse_sensitivities == true && return :ForwardEquations
    end

    if modelSize === :Large
        if "SciMLSensitivity" ∉ string.(values(Base.loaded_modules)) 
            @warn "For large models adjoint sensitivity analysis is the best gradient method. To allow this to be set by default load SciMLSensitivity via using SciMLSensitivity"
            return :ForwardDiff
        end
        return :Adjoint
    end
end


function setHessianMethod(hessian_method::Union{Symbol, Nothing},
                          modelSize::Symbol)

    if !isnothing(hessian_method)
        return hessian_method
    end

    if modelSize === :Small
        return :ForwardDiff
    end

    if modelSize === :Medium || modelSize === :Large
        return :GaussNewton
    end
end


function setSensealg(sensealg,
                     ::Val{:ForwardDiff})
    return nothing
end
function setSensealg(sensealg,
                     ::Val{:ForwardEquations})
    if !isnothing(sensealg)
        @assert sensealg == :ForwardDiff "For gradient method :ForwardEquations allowed sensealg args are :ForwardDiff, ForwardSensitivity(), ForwardDiffSensitivity() not $sensealg"
        return sensealg
    end 
    return :ForwardDiff
end
