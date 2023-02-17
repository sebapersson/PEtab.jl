# PEtab structs
include("PEtab_structs.jl")

include("Common.jl")

# Files related to computing the cost (likelihood)
include(joinpath("Compute_cost", "Compute_priors.jl"))
include(joinpath("Compute_cost", "Compute_cost.jl"))
include(joinpath("Compute_cost", "Compute_cost_zygote.jl"))

# Files related to computing derivatives
include(joinpath("Derivatives", "Hessian.jl"))
include(joinpath("Derivatives", "Gradient.jl"))
include(joinpath("Derivatives", "Adjoint_sensitivity_analysis.jl"))
include(joinpath("Derivatives", "Forward_sensitivity_equations.jl"))
include(joinpath("Derivatives", "Gauss_newton.jl"))
include(joinpath("Derivatives", "Common.jl"))

# Files related to solving the ODE-system
include(joinpath("Solve_ODE", "Change_experimental_condition.jl"))
include(joinpath("Solve_ODE", "Solve_ode_Zygote.jl"))
include(joinpath("Solve_ODE", "Solve_ode_model.jl"))

# Files related to distributed computing
include(joinpath("Distributed", "Distributed.jl"))

# Files related to processing PEtab files
include(joinpath("Process_PEtab_files", "Common.jl"))
include(joinpath("Process_PEtab_files", "Get_simulation_info.jl"))
include(joinpath("Process_PEtab_files", "Get_parameter_indices.jl"))
include(joinpath("Process_PEtab_files", "Process_measurements.jl"))
include(joinpath("Process_PEtab_files", "Process_parameters.jl"))
include(joinpath("Process_PEtab_files", "Process_callbacks.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Common.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Create_h_sigma_derivatives.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Create_u0_h_sigma.jl"))
include(joinpath("Process_PEtab_files", "Read_PEtab_files.jl"))

# For creating a PEtab ODE problem
include(joinpath("Create_PEtab_ODEProblem.jl"))


"""
    setUpPeTabModel(modelName::String, dirModel::String)::PEtabModel
    Given a model directory (dirModel) containing the PeTab files and a
    xml-file on format modelName.xml will return a PEtabModel struct holding
    paths to PeTab files, ode-system in ModellingToolkit format, functions for
    evaluating yMod, u0 and standard deviations, and a parameter and state maps
    for how parameters and states are mapped in the ModellingToolkit ODE system
    along with state and parameter names.
    dirModel must contain a SBML file named modelName.xml, and files starting with
    measurementData, experimentalCondition, parameter, and observables (tsv-files).
    The latter files must be unique (e.g only one file starting with measurementData)
    TODO : Example
"""
function readPEtabModel(pathYAML::String;
                        forceBuildJuliaFiles::Bool=false,
                        verbose::Bool=true,
                        ifElseToEvent::Bool=true,
                        jlFile=false)::PEtabModel

    pathSBML, pathParameters, pathConditions, pathObservables, pathMeasurements, dirJulia, dirModel, modelName = readPEtabYamlFile(pathYAML, jlFile=jlFile)

    if jlFile == false

        pathModelJlFile = joinpath(dirJulia, modelName * ".jl")

        if !isfile(pathModelJlFile) && forceBuildJuliaFiles == false
            verbose == true && @printf("Julia model file does not exist, will build it\n")
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, ifElseToEvent=ifElseToEvent)

        elseif isfile(pathModelJlFile) && forceBuildJuliaFiles == false
            verbose == true && @printf("Julia model file exists at %s - will not rebuild\n", pathModelJlFile)

        elseif forceBuildJuliaFiles == true
            verbose == true && @printf("By user option will rebuild Julia model file\n")
            isfile(pathModelJlFile) == true && rm(pathModelJlFile)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, ifElseToEvent=ifElseToEvent)
        end

    else
        jlDir = joinpath(dirModel, "Julia_model_files")
        modelDict, pathModelJlFile = JLToModellingToolkit(modelName, jlDir, ifElseToEvent=ifElseToEvent)
    end

    # Load model ODE-system
    include(pathModelJlFile)
    expr = Expr(:call, Symbol("getODEModel_" * modelName))
    _odeSystem, stateMap, parameterMap = eval(expr)
    odeSystem = structural_simplify(_odeSystem)
    # TODO : Make these to strings here to save conversions
    parameterNames = parameters(odeSystem)
    stateNames = states(odeSystem)

    # Build functions for observables, sd and u0 if does not exist and include
    path_u0_h_sigma = joinpath(dirJulia, modelName * "_h_sd_u0.jl")
    path_D_h_sd = joinpath(dirJulia, modelName * "_D_h_sd.jl")
    if !isfile(path_u0_h_sigma) || !isfile(path_D_h_sd) || forceBuildJuliaFiles == true
        verbose && forceBuildJuliaFiles == false && @printf("File for h, u0 and σ does not exist will build it\n")
        verbose && forceBuildJuliaFiles == true && @printf("By user option will rebuild h, σ and u0\n")

        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        create_σ_h_u0_File(modelName, pathYAML, dirJulia, odeSystem, stateMap, modelDict, verbose=verbose, jlFile=jlFile)
        createDerivative_σ_h_File(modelName, pathYAML, dirJulia, odeSystem, modelDict, verbose=verbose, jlFile=jlFile)
    else
        verbose == true && @printf("File for h, u0 and σ exists will not rebuild it\n")
    end
    include(path_u0_h_sigma)
    include(path_D_h_sd)

    pathCallback = joinpath(dirJulia, modelName * "_callbacks.jl")
    if !isfile(pathCallback) || forceBuildJuliaFiles == true
        verbose && forceBuildJuliaFiles == false && @printf("File for callback does not exist will build it\n")
        verbose && forceBuildJuliaFiles == true && @printf("By user option will rebuild callback file\n")

        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        createCallbacksForTimeDepedentPiecewise(odeSystem, modelDict, modelName, pathYAML, dirJulia, jlFile = jlFile)
    end
    include(pathCallback)
    exprCallback = Expr(:call, Symbol("getCallbacks_" * modelName))
    cbSet::CallbackSet, checkCbActive::Vector{Function} = eval(exprCallback)

    # Check if callbacks are triggered at a time-point defined by a model parameter. If true then to accurately compute
    # the gradient tsops must be a vector of floats, and the timespan of the ODE-problem must be converted to Duals.
    if !@isdefined(modelDict)
        modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
    end
    convertTspan = shouldConvertTspan(pathYAML, modelDict, odeSystem, jlFile)

    petabModel = PEtabModel(modelName,
                            compute_h,
                            compute_u0!,
                            compute_u0,
                            compute_σ,
                            compute_∂h∂u!,
                            compute_∂σ∂σu!,
                            compute_∂h∂p!,
                            compute_∂σ∂σp!,
                            computeTstops,
                            convertTspan,
                            odeSystem,
                            parameterMap,
                            stateMap,
                            parameterNames,
                            stateNames,
                            dirModel,
                            dirJulia,
                            pathMeasurements,
                            pathConditions,
                            pathObservables,
                            pathParameters,
                            pathSBML,
                            pathYAML,
                            cbSet,
                            checkCbActive)

    return petabModel
end
