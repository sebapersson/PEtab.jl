function _logging(whatlog::Symbol, verbose::Bool; time = nothing, name::String = "",
                  buildfiles::Bool = false, exist::Bool = false,
                  method::Union{Symbol, String} = "")::Nothing
    verbose == false && return nothing

    if !isnothing(time)
        str = @sprintf("done. Time = %.1es\n", time)
        print(str)
        return nothing
    end

    # For PEtabModel
    if whatlog == :Build_PEtabModel
        str = styled"{blue:{bold:┌Info:}} Building {magenta:PEtabModel} for model $(name)\n"
    end
    if whatlog == :Build_SBML
        if buildfiles == true && exist == true
            str = styled"{blue:{bold:│ }} By user option reimports {magenta:SBML} model ... "
        end
        if exist == false
            str = styled"{blue:{bold:│ }} Imports {magenta:SBML} model ... "
        end
        if buildfiles == false && exist == true
            str = styled"{blue:{bold:│ }} {magenta:SBML} model already imported\n"
        end
    end
    if whatlog == :Build_ODESystem
        str = styled"{blue:{bold:│ }} Parses the SBML model into an {magenta:ODESystem} ... "
    end
    if whatlog == :Build_u0_h_σ
        if buildfiles == true && exist == true
            str = styled"{blue:{bold:│ }} By user option rebuilds {magenta:u0}, " *
                  styled"{magenta:h} and {magenta:σ} functions ... "
        end
        if exist == false
            str = styled"{blue:{bold:│ }} Builds {magenta:u0}, {magenta:h} and " *
                  styled"{magenta:σ} functions ... "
        end
        if buildfiles == false && exist == true
            str = styled"{blue:{bold:│ }} {magenta:u0}, {magenta:h} and {magenta:σ} " *
                  styled"functions already exist\n"
        end
    end
    if whatlog == :Build_∂_h_σ
        if buildfiles == true && exist == true
            str = styled"{blue:{bold:│ }} By user option rebuilds {magenta:∂h∂p}, " *
                  styled"{magenta:∂h∂u}, {magenta:∂σ∂p} and {magenta:∂σ∂u} functions ... "
        end
        if exist == false
            str = styled"{blue:{bold:│ }} Builds {magenta:∂h∂p}, {magenta:∂h∂u}, " *
                  styled"{magenta:∂σ∂p} and {magenta:∂σ∂u} functions ... "
        end
        if buildfiles == false && exist == true
            str = styled"{blue:{bold:│ }} {magenta:∂h∂p}, {magenta:∂h∂u}, " *
                  styled"{magenta:∂σ∂p} and {magenta:∂σ∂u} functions already exist\n"
        end
    end
    if whatlog == :Build_callbacks
        str = styled"{blue:{bold:└ }} Builds {magenta:callback} (events) functions ... "
    end

    # For PEtabODEProblem
    if whatlog == :Build_PEtabODEProblem
        str = styled"{blue:{bold:┌Info:}} Building {magenta:PEtabODEProblem} for model $(name)\n"
    end
    if whatlog == :Build_ODEProblem
        str = styled"{blue:{bold:│ }} Building {magenta:ODEProblem} from model system ... "
    end
    if whatlog == :Build_nllh
        str = styled"{blue:{bold:│ }} Building {magenta:nllh} (negative log-likelihood) function ... "
    end
    if whatlog == :Build_gradient
        str = styled"{blue:{bold:│ }} Building {magenta:gradient} nllh function for method $(method) ... "
    end
    if whatlog == :Build_hessian
        str = styled"{blue:{bold:└ }} Building {magenta:Hessian} nllh function for method $(method) ... "
    end

    print(str)
    return nothing
end
