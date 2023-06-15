#=
    Helper functions for setting default values when creating a PEtabODEProblem in case left unspecifed by the
    user.
=#


function setODESolverOptions(odeSolverOptions::Union{ODESolverOptions, Nothing},
                             modelSize::Symbol,
                             gradientMethod::Symbol)::ODESolverOptions

    if !isnothing(odeSolverOptions)
        return odeSolverOptions
    end

    if modelSize === :Small
        return ODESolverOptions(Rodas5P())
    end

    if modelSize === :Medium
        return ODESolverOptions(QNDF())
    end

    if modelSize === :Large
        @warn "For large models we strongly recomend to compare different ODE-solvers instead of using default options"
        if gradientMethod === :Adjoint || isnothing(gradientMethod)
            return ODESolverOptions(CVODE_BDF())
        else
            return ODESolverOptions(KenCarp4())
        end
    end
end


function setSteadyStateSolverOptions(ssOptions::Union{SteadyStateSolverOptions, Nothing},
                                     odeSolverOptions::ODESolverOptions)::SteadyStateSolverOptions

    if !isnothing(ssOptions)
        return ssOptions
    end

     ssSolverOptions = SteadyStateSolverOptions(:Simulate,
                                                abstol=odeSolverOptions.abstol / 100,
                                                reltol=odeSolverOptions.reltol / 100)
    return ssSolverOptions
end



function setGradientMethod(gradientMethod::Union{Symbol, Nothing},
                           modelSize::Symbol,
                           reuseS::Bool)::Symbol

    if !isnothing(gradientMethod)
        return gradientMethod
    end

    if modelSize === :Small
        return :ForwardDiff
    end

    if modelSize === :Medium
        reuseS == false && return :ForwardDiff
        reuseS == true && return :ForwardEquations
    end

    if modelSize === :Large
        if "SciMLSensitivity" ∉ string.(values(Base.loaded_modules)) 
            @warn "For large models adjoint sensitivity analysis is the best gradient method. To allow this to be set by default load SciMLSensitivity via using SciMLSensitivity"
            return :ForwardDiff
        end
        return :Adjoint
    end
end


function setHessianMethod(hessianMethod::Union{Symbol, Nothing},
                          modelSize::Symbol)

    if !isnothing(hessianMethod)
        return hessianMethod
    end

    if modelSize === :Small
        return :ForwardDiff
    end

    if modelSize === :Medium || modelSize === :Large
        return :GaussNewton
    end
end


function setSensealg(sensealg,
                     gradientMethod)

    # Sanity check user gradient input
    if !isnothing(sensealg)
        if gradientMethod === :ForwardEquations
            @assert sensealg == :ForwardDiff || any(typeof(sensealg) .<: [ForwardSensitivity, ForwardDiffSensitivity]) "For gradient method :ForwardEquations allowed sensealg args are :ForwardDiff, ForwardSensitivity(), ForwardDiffSensitivity() not $sensealg"
        end
        
        if gradientMethod === :Zygote
            @assert (typeof(sensealg) <: SciMLSensitivity.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
        end
        return sensealg
    end

    if gradientMethod === :ForwardDiff || gradientMethod === :ForwardEquations
        return :ForwardDiff
    end
end
