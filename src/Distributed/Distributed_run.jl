function runProcess(jobs, results) 
    
    # Import actual ODE model
    odeProblem::ODEProblem = take!(jobs)[1]
    put!(results, tuple(:Done))

    # Import structs needed to compute the cost, gradient, and hessian
    petabModel::PEtabModel = take!(jobs)[1]
    put!(results, tuple(:Done))
    parameterInfo::ParametersInfo = take!(jobs)[1]
    put!(results, tuple(:Done))
    measurementInfo::MeasurementsInfo = take!(jobs)[1]
    put!(results, tuple(:Done))
    simulationInfo::SimulationInfo = take!(jobs)[1]
    put!(results, tuple(:Done))
    θ_indices::ParameterIndices = take!(jobs)[1]
    put!(results, tuple(:Done))
    priorInfo::PriorInfo = take!(jobs)[1]
    put!(results, tuple(:Done))
    println("Done loading structs for ", myid())

    odeSolver, solverAbsTol::Float64, solverRelTol::Float64 = take!(jobs)
    put!(results, tuple(:Done))
    odeSolverAdjoint, solverAdjointAbsTol::Float64, solverAdjointRelTol::Float64, sensealgAdjoint, sensealgAdjointSS = take!(jobs)
    put!(results, tuple(:Done))
    odeSolverForwardEquations, sensealgForwardEquations, chunkSize = take!(jobs)
    put!(results, tuple(:Done))
    println("Done loading solver and gradient options for ", myid())

    expIDs::Vector{Symbol} = take!(jobs)[1]

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations.
    # Due to its slow performance we do not support Zygote 
    computeCost = setUpCost(:Standard, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, 
                            simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, expIDs)                      

    odeProblemForwardEquations = getODEProblemForwardEquations(odeProblem, sensealgForwardEquations)                            
    computeGradientAutoDiff = setUpGradient(:AutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, 
                                            simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, expIDs, 
                                            chunkSize=chunkSize)
    computeGradientForwardEquations = setUpGradient(:ForwardEquations, odeProblemForwardEquations, odeSolverForwardEquations, solverAbsTol, 
                                                    solverRelTol, petabModel, simulationInfo, θ_indices, measurementInfo, 
                                                    parameterInfo, priorInfo, expIDs, sensealg=sensealgForwardEquations, chunkSize=chunkSize)                                                   
    computeGradientAdjoint = setUpGradient(:Adjoint, odeProblem, odeSolverAdjoint, solverAdjointAbsTol, solverAdjointRelTol, 
                                           petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, 
                                           expIDs, sensealg=sensealgAdjoint, sensealgSS=sensealgAdjointSS)   

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the 
    # Gauss-Newton method.                                        
    computeHessian = setUpHessian(:AutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, simulationInfo,
                                  θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize, expIDs)
    computeHessianBlock = setUpHessian(:BlockAutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, simulationInfo,
                                        θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize, expIDs)                                  
    computeHessianGN = setUpHessian(:GaussNewton, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, simulationInfo,
                                    θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize, expIDs, reuseS=false)                                                                          

    put!(results, tuple(:Done))  
    gradient = zeros(Float64, length(θ_indices.θ_estNames))
    hessian = zeros(Float64, length(θ_indices.θ_estNames), length(θ_indices.θ_estNames))

    println("Done setting up cost, gradient and hessian functions for ", myid())

    while true 
        θ_est::Vector{Float64}, task::Symbol = take!(jobs)
        if task == :Cost
            cost = computeCost(θ_est)
            put!(results, tuple(:Done, cost))
        end
        
        if task == :AutoDiff
            computeGradientAutoDiff(gradient, θ_est)
            put!(results, tuple(:Done, gradient))
        end

        if task == :Adjoint
            computeGradientAdjoint(gradient, θ_est)
            put!(results, tuple(:Done, gradient))
        end

        if task == :ForwardEquations
            computeGradientForwardEquations(gradient, θ_est)
            put!(results, tuple(:Done, gradient))
        end

        if task == :BlockAutoDiff
            computeHessianBlock(hessian, θ_est)
            put!(results, tuple(:Done, hessian))
        end

        if task == :HessianAutoDiff
            computeHessian(hessian, θ_est)
            put!(results, tuple(:Done, hessian))
        end

        if task == :GaussNewton
            computeHessianGN(hessian, θ_est)
            put!(results, tuple(:Done, hessian))
        end
    end
end