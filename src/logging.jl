StyledStrings.addface!(:PURPLE => StyledStrings.Face(foreground = 0x8f4093))

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
        str = styled"{PURPLE:{bold:┌Info:}} Building {emphasis:PEtabModel} for model $(name)\n"
    end
    if whatlog == :Build_SBML
        if buildfiles == true && exist == true
            str = styled"{PURPLE:{bold:│ }} By user option reimports {emphasis:SBML} model ... "
        end
        if exist == false
            str = styled"{PURPLE:{bold:│ }} Imports {emphasis:SBML} model ... "
        end
        if buildfiles == false && exist == true
            str = styled"{PURPLE:{bold:│ }} {emphasis:SBML} model already imported\n"
        end
    end
    if whatlog == :Build_ODESystem
        str = styled"{PURPLE:{bold:│ }} Parses the SBML model into an {emphasis:ODESystem} ... "
    end
    if whatlog == :Build_u0_h_σ
        if buildfiles == true && exist == true
            str = styled"{PURPLE:{bold:│ }} By user option rebuilds {emphasis:u0}, " *
                  styled"{emphasis:h} and {emphasis:σ} functions ... "
        end
        if exist == false
            str = styled"{PURPLE:{bold:│ }} Building {emphasis:u0}, {emphasis:h} and \
                         {emphasis:σ} functions ... "
        end
        if buildfiles == false && exist == true
            str = styled"{PURPLE:{bold:│ }} {emphasis:u0}, {emphasis:h} and {emphasis:σ} " *
                  styled"functions already exist\n"
        end
    end
    if whatlog == :Build_∂_h_σ
        if buildfiles == true && exist == true
            str = styled"{PURPLE:{bold:│ }} By user option rebuilds {emphasis:∂h∂p}, " *
                  styled"{emphasis:∂h∂u}, {emphasis:∂σ∂p} and {emphasis:∂σ∂u} functions ... "
        end
        if exist == false
            str = styled"{PURPLE:{bold:│ }} Building {emphasis:∂h∂p}, {emphasis:∂h∂u}, " *
                  styled"{emphasis:∂σ∂p} and {emphasis:∂σ∂u} functions ... "
        end
        if buildfiles == false && exist == true
            str = styled"{PURPLE:{bold:│ }} {emphasis:∂h∂p}, {emphasis:∂h∂u}, " *
                  styled"{emphasis:∂σ∂p} and {emphasis:∂σ∂u} functions already exist\n"
        end
    end
    if whatlog == :Build_callbacks
        str = styled"{PURPLE:{bold:└ }} Building {emphasis:callback} (events) functions ... "
    end

    # For PEtabODEProblem
    if whatlog == :Build_PEtabODEProblem
        str = styled"{PURPLE:{bold:┌Info:}} Building {emphasis:PEtabODEProblem} for model $(name)\n"
    end
    if whatlog == :Build_PEtabSDEProblem
        str = styled"{PURPLE:{bold:┌Info:}} Building {emphasis:PEtabSDEProblem} for model $(name)\n"
    end
    if whatlog == :Build_ODEProblem
        str = styled"{PURPLE:{bold:│ }} Building {emphasis:ODEProblem} from model system ... "
    end
    if whatlog == :Build_SDEProblem
        str = styled"{PURPLE:{bold:└ }} Building {emphasis:SDEProblem} from model system ... "
    end
    if whatlog == :Build_nllh
        str = styled"{PURPLE:{bold:│ }} Building {emphasis:nllh} (negative log-likelihood) function ... "
    end
    if whatlog == :Build_gradient
        str = styled"{PURPLE:{bold:│ }} Building {emphasis:gradient} nllh function for method $(method) ... "
    end
    if whatlog == :Build_hessian
        str = styled"{PURPLE:{bold:│ }} Building {emphasis:Hessian} nllh function for method $(method) ... "
    end
    if whatlog == :Build_chi2_res_sim
        str = styled"{PURPLE:{bold:└ }} Building {emphasis:χ₂} and {emphasis:residuals} functions ... "
    end

    print(str)
    return nothing
end
