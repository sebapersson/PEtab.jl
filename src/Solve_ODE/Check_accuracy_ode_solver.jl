"""
    calcHighAccOdeSolution(prob::ODEProblem,
                           changeToExperimentalCondUse!::Function,
                           measurementData::DataFrame,
                           simulationInfo::SimulationInfo,
                           tol::Float64=1e-15)

    For a PeTab ODE model with the parameter values in prob compute a high accuracy ODE solution
    with absTol=relTol=tol using BigFloats for the ode-problem. The ODE is solved for all experimental
    conditions specified in the PeTab files. Returns an array with ODE-solution and a bool which is
    true if the ODE model could be solved.

    By default the function first tries to solve the ODE problem using a non-stiff solver AutoVern9(Rodas4P()).
    In case the non-stiff solver fails the Rodas4P() is used to compute the high accuracy solution. If both solvers
    fail status fail is returned.
"""
function computeHighAccuracyOdeSolution(odeProblem::ODEProblem,
                                        changeExperimentalCondition!::Function,
                                        simulationInfo::SimulationInfo,
                                        computeTStops::Function;
                                        absTol::Float64=1e-15,
                                        relTol::Float64=1e-15,
                                        nTimePointsSave=100)

    bigFloatODEProblem = createBigFloatODEProblem(odeProblem)

    solverNonStiff = AutoVern9(Rodas4P())
    solverStiff = Rodas4P()
    local highAccuracySolutions
    local sucessSolver
    try
        highAccuracySolutions, sucessSolver = solveODEAllExperimentalConditions(bigFloatODEProblem, changeExperimentalCondition!, simulationInfo, solverNonStiff, absTol, relTol, computeTStops; nTimePointsSave=nTimePointsSave)
    catch
        highAccuracySolutions, sucessSolver = solveODEAllExperimentalConditions(bigFloatODEProblem, changeExperimentalCondition!, simulationInfo, solverStiff, absTol, relTol, computeTStops; nTimePointsSave=nTimePointsSave)
    end
    GC.gc()

    # In cases the non-stiff solver just fails but does not produce an error 
    if sucessSolver != true
        println("Failed with composite solver - moving on to stiff solver")
        try
            highAccuracySolutions, sucessSolver = solveODEAllExperimentalConditions(bigFloatODEProblem, changeExperimentalCondition!, simulationInfo, solverStiff, absTol, relTol, computeTStops; nTimePointsSave=nTimePointsSave)
        catch
            sucessSolver = false
            highAccuracySolutions = nothing
        end
    end

    return highAccuracySolutions, sucessSolver
end


"""
    computeAccuracyODESolver(prob::ODEProblem,
                             solArrayHighAccuracy::Array{Union{OrdinaryDiffEq.ODECompositeSolution, ODESolution}, 1},
                            changeToExperimentalCondUse!::Function,
                          measurementInfo::DataFrame,
                          simulationInfo::SimulationInfo,
                          solver,
                          tol::Float64)::Float64

    Check the accuracy of an ODE solver at specific tol=absTol=relTol for a PeTab ODE model (odeProb) by
    for each experimental condition computing the squared sum difference against a high accuracy ODE
    solution (for each experimental condition) stored in solArrayHighAccuracy.

    Recomended to compute high accuracy solution with small tolerances (1e-15) using a high accuracy solver
    and BigFloat.
"""
function computeAccuracyODESolver(odeProblem::ODEProblem,
                                  highAccuracySolutions,
                                  changeExperimentalCondition!::Function,
                                  simulationInfo::SimulationInfo,
                                  solver,
                                  absTol::Float64,
                                  relTol::Float64,
                                  computeTStops::Function)::Float64

    # Check if model can be solved (without using forced stops for integrator
    # as this can make the solver converge).
    __tmp, sucess = solveODEAllExperimentalConditions(odeProblem, changeExperimentalCondition!, simulationInfo, solver, absTol, relTol, computeTStops; denseSolution=false)
    __tmp = 0
    GC.gc()
    if sucess == false
        return Inf
    end

    # Compute the square error by comparing the solver against the high accuracy solArray. The comparison
    # is done at the time-points in the high accuracy solver.
    sqErr::Float64 = 0.0

    # We only compute accuracy for post-equlibrium models
    if simulationInfo.haspreEquilibrationConditionId == true

        # Arrays to store steady state (pre-eq) values.
        preEquilibrationId = unique(simulationInfo.preEquilibrationConditionId)
        uAtSS = Matrix{eltype(odeProblem.p)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))
        u0AtT0 = Matrix{eltype(odeProblem.p)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))

        for i in eachindex(preEquilibrationId)
            _odeSolutions = simulationInfo.odePreEqulibriumSolutions
            _odeSolutions[preEquilibrationId[i]] = solveODEPreEqulibrium!((@view uAtSS[:, i]),
                                                                          (@view u0AtT0[:, i]),
                                                                          odeProblem,
                                                                          changeExperimentalCondition!,
                                                                          preEquilibrationId[i],
                                                                          absTol,
                                                                          relTol,
                                                                          solver,
                                                                          simulationInfo.callbackSS,
                                                                          false)
        end
    end

    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalId = simulationInfo.experimentalConditionId[i]
        highAccuracySolution = highAccuracySolutions[experimentalId]

        # Sanity check and process user input.
        tMax = simulationInfo.timeMax[experimentalId]

        # In case we have a simulation with PreEqulibrium
        if simulationInfo.preEquilibrationConditionId[i] != :None
            whichIndex = findfirst(x -> x == simulationInfo.preEquilibrationConditionId[i], preEquilibrationId)
            odeSolution = solveODEPostEqulibrium(odeProblem,
                                                 (@view uAtSS[:, whichIndex]),
                                                 (@view u0AtT0[:, whichIndex]),
                                                 changeExperimentalCondition!,
                                                 simulationInfo,
                                                 simulationInfo.simulationConditionId[i],
                                                 experimentalId,
                                                 absTol,
                                                 relTol,
                                                 tMax,
                                                 solver,
                                                 computeTStops,
                                                 denseSolution=false,
                                                 tSave=highAccuracySolution.t)

        # In case we have an ODE solution without Pre-equlibrium
        else
            odeSolution = solveODENoPreEqulibrium!(odeProblem,
                                                   changeExperimentalCondition!,
                                                   simulationInfo,
                                                   simulationInfo.simulationConditionId[i],
                                                   absTol,
                                                   relTol,
                                                   solver,
                                                   tMax,
                                                   computeTStops,
                                                   denseSolution=false,
                                                   tSave=highAccuracySolution.t)

        end

        sqErr += calcSqErrVal(highAccuracySolution, odeSolution, tMax)
    end

    return sqErr
end


"""
    calcSqErrVal(solHighAccuracy, sol, t_max::Float64)::Float64

    Helper function to compute the squared sum difference betweeen two
    ODESolutions solved to t_max. Here solHighAccuracy is meant to
    be a high accuracy solution.
"""
function calcSqErrVal(solHighAccuracy, sol, t_max::Float64)::Float64

    sqErr::Float64 = 0.0

    # If t_max = Inf only compare at the end-point (steady state value)
    if isinf(t_max)
        sqErr = sum((solHighAccuracy.u[end] .- sol.u[end]).^2)

    # In case solHighAccuracy and sol do not have equal number of time-points
    # there are events in the solution causing problem (events forces saving).
    # Hence sol is interpolated to solHighAccuracy values, and to avoid interpolation
    # errors around events the even sqErr is subtracted when solHighAccuracy is interpolated
    # upon itself.
    elseif length(solHighAccuracy.t) != length(sol.t)
        sqErr = sum((Array(sol(solHighAccuracy.t)) - Array(solHighAccuracy)).^2)
        sqErr -= sum((Array(solHighAccuracy(solHighAccuracy.t)) - Array(solHighAccuracy)).^2)

    # Same strategy as above to avoid interpolation errors with events.
    else
        sqErr = sum((sol[:,:] - solHighAccuracy[:,:]).^2)
        sqErr -= sum((Array(solHighAccuracy(solHighAccuracy.t)) - Array(solHighAccuracy)).^2)
    end

    return sqErr
end


"""
    createBigFloatODEProblem(petabModel::PEtabModel)::ODEProblem

    From a PeTab model create its corresponding ODE-problem with
    BigFloat (long double).
"""
function createBigFloatODEProblem(petabModel::PEtabModel)::ODEProblem

    parameterMapUse = convert(Vector{Pair{Num, BigFloat}}, petabModel.parameterMap)

    if typeof(petabModel.stateMap) == Vector{Pair{Num, Num}}
        # This means that u0 has some initializations that depend on par,
        # since the values of par is BigFloat, u0 will be BigFloat
        stateMapUse = petabModel.stateMap
    elseif typeof(petabModel.stateMap) == Vector{Pair{Num, Float64}}
        stateMapUse = convert(Vector{Pair{Num, BigFloat}}, petabModel.stateMap)
    else
        println("Error: Could not parse PeTab model into BigFloat problem")
    end

    prob = ODEProblem(petabModel.odeSystem, stateMapUse, (0.0, 1e8), parameterMapUse, jac=true)
    return prob
end
"""
    createBigFloatODEProblem(odeProb::ODEProblem)::ODEProblem

    Convert an ODE problem with arbitrary float to one an ODE-problem with BigFloat.
"""
function createBigFloatODEProblem(odeProb::ODEProblem)::ODEProblem
    return remake(odeProb, p = convert.(BigFloat, odeProb.p), u0 = convert.(BigFloat, odeProb.u0))
end
