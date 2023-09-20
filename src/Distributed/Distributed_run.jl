function runProcess(jobs, results)

    # Import actual ODE model
    odeProblem::ODEProblem = take!(jobs)[1]
    put!(results, tuple(:Done))

    # Import structs needed to compute the cost, gradient, and hessian
    petab_model::PEtabModel = take!(jobs)[1]
    put!(results, tuple(:Done))
    parameterInfo::ParametersInfo = take!(jobs)[1]
    put!(results, tuple(:Done))
    measurementInfo::MeasurementsInfo = take!(jobs)[1]
    put!(results, tuple(:Done))
    simulation_info::SimulationInfo = take!(jobs)[1]
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
    odeSolverForwardEquations, sensealgForwardEquations, chunksize = take!(jobs)
    put!(results, tuple(:Done))
    println("Done loading solver and gradient options for ", myid())

    expIDs::Vector{Symbol} = take!(jobs)[1]

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations.
    # Due to its slow performance we do not support Zygote
    compute_cost = setUpCost(:Standard, odeProblem, odeSolver, solverAbsTol, solverRelTol, petab_model,
                            simulation_info, θ_indices, measurementInfo, parameterInfo, priorInfo, expIDs)

    odeProblemForwardEquations = getODEProblemForwardEquations(odeProblem, sensealgForwardEquations)
    compute_gradientAutoDiff = setUpGradient(:AutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petab_model,
                                            simulation_info, θ_indices, measurementInfo, parameterInfo, priorInfo, expIDs,
                                            chunksize=chunksize)
    compute_gradientForwardEquations = setUpGradient(:ForwardEquations, odeProblemForwardEquations, odeSolverForwardEquations, solverAbsTol,
                                                    solverRelTol, petab_model, simulation_info, θ_indices, measurementInfo,
                                                    parameterInfo, priorInfo, expIDs, sensealg=sensealgForwardEquations, chunksize=chunksize)
    compute_gradientAdjoint = setUpGradient(:Adjoint, odeProblem, odeSolverAdjoint, solverAdjointAbsTol, solverAdjointRelTol,
                                           petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                           expIDs, sensealg=sensealgAdjoint, sensealg_ss=sensealgAdjointSS)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss-Newton method.
    compute_hessian = setUpHessian(:AutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petab_model, simulation_info,
                                  θ_indices, measurementInfo, parameterInfo, priorInfo, chunksize, expIDs)
    compute_hessianBlock = setUpHessian(:BlockAutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petab_model, simulation_info,
                                        θ_indices, measurementInfo, parameterInfo, priorInfo, chunksize, expIDs)
    compute_hessianGN = setUpHessian(:GaussNewton, odeProblem, odeSolver, solverAbsTol, solverRelTol, petab_model, simulation_info,
                                    θ_indices, measurementInfo, parameterInfo, priorInfo, chunksize, expIDs, reuse_sensitivities=false)

    put!(results, tuple(:Done))
    gradient = zeros(Float64, length(θ_indices.θ_names))
    hessian = zeros(Float64, length(θ_indices.θ_names), length(θ_indices.θ_names))

    println("Done setting up cost, gradient and hessian functions for ", myid())

    while true
        θ_est::Vector{Float64}, task::Symbol = take!(jobs)
        if task == :Cost
            cost = compute_cost(θ_est)
            put!(results, tuple(:Done, cost))
        end

        if task == :AutoDiff
            compute_gradientAutoDiff(gradient, θ_est)
            put!(results, tuple(:Done, gradient))
        end

        if task == :Adjoint
            compute_gradientAdjoint(gradient, θ_est)
            put!(results, tuple(:Done, gradient))
        end

        if task == :ForwardEquations
            compute_gradientForwardEquations(gradient, θ_est)
            put!(results, tuple(:Done, gradient))
        end

        if task == :BlockAutoDiff
            compute_hessianBlock(hessian, θ_est)
            put!(results, tuple(:Done, hessian))
        end

        if task == :HessianAutoDiff
            compute_hessian(hessian, θ_est)
            put!(results, tuple(:Done, hessian))
        end

        if task == :GaussNewton
            compute_hessianGN(hessian, θ_est)
            put!(results, tuple(:Done, hessian))
        end
    end
end
