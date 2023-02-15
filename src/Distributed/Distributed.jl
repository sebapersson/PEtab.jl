using Distributed


function setUpProcesses(petabModel::PEtabModel,
                        odeSolver::SciMLAlgorithm, 
                        solverAbsTol::Float64, 
                        solverRelTol::Float64,
                        odeSolverAdjoint::SciMLAlgorithm,
                        sensealgAdjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                        sensealgAdjointSS::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                        solverAdjointAbsTol::Float64,
                        solverAdjointRelTol::Float64,
                        odeSolverForwardEquations::SciMLAlgorithm, 
                        sensealgForwardEquations::Union{Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm},
                        parameterInfo::ParametersInfo,
                        measurementInfo::MeasurementsInfo, 
                        simulationInfo::SimulationInfo, 
                        θ_indices::ParameterIndices, 
                        pirorInfo::PriorInfo,
                        odeProblem::ODEProblem, 
                        chunkSize::Union{Int64, Nothing})                              

    println("Setting up cost, grad, and hessian to be computed on several processes using Distributed.jl")

    # Make functions, structs and packages aware for each process 
    loadPackages()
    loadFunctionsAndStructs()
    loadYmodSdU0(petabModel)
    
    nProcs = nprocs()                                  
    experimentalConditionId = unique(simulationInfo.experimentalConditionId)
    if nProcs > length(experimentalConditionId)
        println("Warning - There are less experimental conditions than processes. Hence some processes will run empty")
        println("Number of processes = $nProcs, number of experimental conditions = ", length(simulationInfo.conditionIdSol))
    end
    idsEachProcess = collect(Iterators.partition(experimentalConditionId, Int(round(length(experimentalConditionId) /nProcs))))

    # Divide the experimental conditions between the number of processes and set up channels for 
    # communicating with each process
    jobs = [RemoteChannel(()->Channel{Tuple}(1)) for i in 1:nProcs]
    results = [RemoteChannel(()->Channel{Tuple}(1)) for i in 1:nProcs]

    # Send ODE-problem, and simultaneously launch the processes 
    for i in 1:nProcs
        @async put!(jobs[i], tuple(deepcopy(odeProblem))) 
        remote_do(runProcess, procs()[i], jobs[i], results[i])
        status = take!(results[i])[1]
        if status != :Done
            println("Error : Could not send ODEProblem to proces ", procs()[i])
        end
    end

    # Send required PEtab structs to processes 
    for i in 1:nProcs
        sendPEtabStruct(petabModel, jobs[i], results[i], "PEtab model", procs()[i])
        sendPEtabStruct(parameterInfo, jobs[i], results[i], "Parameter data", procs()[i])
        sendPEtabStruct(measurementInfo, jobs[i], results[i], "Measurement data", procs()[i])
        sendPEtabStruct(simulationInfo, jobs[i], results[i], "Simulation info", procs()[i])
        sendPEtabStruct(θ_indices, jobs[i], results[i], "Parameter indices", procs()[i])
        sendPEtabStruct(pirorInfo, jobs[i], results[i], "Prior info", procs()[i])
    end

    # Send solver and solver tolerance 
    for i in 1:nProcs
        @async put!(jobs[i], tuple(odeSolver, solverAbsTol, solverRelTol)) 
        status = take!(results[i])[1]
        if status != :Done
            println("Error : Could not send solver and tolerance problem to proces ", procs()[i])
        end
    end

    # Send adjoint solver, adjont tolerance and adjoint solver tolerance
    for i in 1:nProcs
        @async put!(jobs[i], tuple(odeSolverAdjoint, solverAdjointAbsTol, solverAdjointRelTol, sensealgAdjoint, sensealgAdjointSS)) 
        status = take!(results[i])[1]
        if status != :Done
            println("Error : Could not send adjoint solver info to process ", procs()[i])
        end
    end

    # Send forward sensitivity equations solver and associated 
    for i in 1:nProcs
        @async put!(jobs[i], tuple(odeSolverForwardEquations, sensealgForwardEquations, chunkSize)) 
        status = take!(results[i])[1]
        if status != :Done
            println("Error : Could not send adjSolver, adjTolerance and adjSensealg to proces ", procs()[i])
        end
    end

    # Send experimental ID:s process each process works with 
    for i in 1:nProcs
        @async put!(jobs[i], tuple(idsEachProcess[i])) 
        status = take!(results[i])[1]
        if status != :Done
            println("Error : Could not send ExpIds problem to proces ", procs()[i])
        end
    end

    return jobs, results 
end


function sendPEtabStruct(structSend::Union{PEtabModel, ParametersInfo, MeasurementsInfo, SimulationInfo, ParameterIndices, PriorInfo}, 
                         job::RemoteChannel, 
                         result::RemoteChannel, 
                         strSend::String, 
                         pID::Integer)

    @async put!(job, tuple(deepcopy(structSend)))   
    status = take!(result)[1]                     
    if status != :Done
        println("Error : Could not send $strSend to process ", pID)
    end
end


function removeAllProcs()
    if length(procs()) > 1
        procsAvailble = procs()[2:end]
        rmprocs(procsAvailble)
    end
    return 
end


function loadPackages()
    print("Loading required packages for each process ... ")
    @eval @everywhere begin 
                        macro LoadLib()
                            quote
                                using DifferentialEquations
                                using ModelingToolkit
                                using DataFrames
                                using LinearAlgebra
                                using ForwardDiff
                                using ReverseDiff
                                using Zygote
                                using Printf
                                using SciMLSensitivity
                            end
                        end
                    end
    @eval @everywhere @LoadLib()
    print("done \n")
end
 

function loadFunctionsAndStructs()
    print("Loading required functions and structs for each process ... ")
    @eval @everywhere begin 
                        macro LoadFuncStruct()
                            quote
                                include(joinpath(pwd(), "src", "Create_PEtab_model.jl"))
                                include(joinpath(pwd(), "src", "Distributed", "Distributed_run.jl"))
                            end
                        end
                    end
    @eval @everywhere @LoadFuncStruct()
    print("done \n")
end
 

function loadYmodSdU0(petabModel::PEtabModel)
    print("Loading u0, σ, h and callback functions ... ")
    path_u0_h_sigma = joinpath(petabModel.dirJulia, petabModel.modelName * "_h_sd_u0.jl")
    path_D_h_sd = joinpath(petabModel.dirJulia, petabModel.modelName * "_D_h_sd.jl")
    pathCallback = joinpath(petabModel.dirJulia, petabModel.modelName * "_callbacks.jl")
    @eval @everywhere include($path_u0_h_sigma)
    @eval @everywhere include($path_D_h_sd)
    @eval @everywhere include($pathCallback)
    print("done \n")
end

