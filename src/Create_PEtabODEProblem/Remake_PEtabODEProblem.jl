"""
    remakePEtabProblem(petabProblem::PEtabODEProblem, parametersChange::Dict) :: PEtabODEProblem

Fixate model parameters for a given PEtabODEProblem without recompiling the problem.

This function allows you to modify parameters without the need to recompile the underlying code, resulting in reduced 
latency. To fixate the parameter k1, you can use `parametersChange=Dict(:k1 => 1.0)`.

If model derivatives are computed using ForwardDiff.jl with a chunk-size of N, the new PEtabODEProblem will only 
evaluate the necessary number of chunks of size N to compute the full gradient for the remade problem. 
"""
function remakePEtabProblem(petabProblem::PEtabODEProblem, parametersChange::Dict)::PEtabODEProblem

    # Only keep which parameters should be fixed 
    for key in keys(parametersChange)
        if parametersChange[key] == "estimate"
            key ∉ petabProblem.θ_estNames && @error "When remaking an PEtab problem we cannot set new parameters in addition to those in the PEtab-file to be estimated"
            delete!(parametersChange, key)
        end
    end

    parametersFix = collect(keys(parametersChange))
    iParametersFix = [findfirst(x -> x == parameterFix, petabProblem.θ_estNames) for parameterFix in parametersFix]
    parametersFixValues = Vector{Float64}(undef, length(parametersFix))
    # Ensure we fixate parameter values on the correct scale 
    for i in eachindex(iParametersFix)
        transform = petabProblem.computeCost.parameterInfo.parameterScale[findfirst(x -> x == parametersFix[i], petabProblem.computeCost.parameterInfo.parameterId)]
        if transform === :lin
            parametersFixValues[i] = parametersChange[parametersFix[i]]
        elseif transform === :log
            parametersFixValues[i] = log(parametersChange[parametersFix[i]])
        elseif transform === :log10
            parametersFixValues[i] = log10(parametersChange[parametersFix[i]])
        end
    end

    # Setup parameters for new problem of lower dimension 
    iUse = findall(x -> x ∉ parametersFix, petabProblem.θ_estNames)
    lowerBounds = petabProblem.lowerBounds[iUse]
    upperBounds = petabProblem.upperBounds[iUse]
    θ_estNames = petabProblem.θ_estNames[iUse]
    θ_nominal = petabProblem.θ_nominal[iUse]
    θ_nominalT = petabProblem.θ_nominalT[iUse]

    # Gradient place-holders for the underlaying functions 
    _θ_est::Vector{Float64} = similar(petabProblem.lowerBounds)
    _gradient::Vector{Float64} = similar(_θ_est)
    _hessian::Matrix{Float64} = Matrix{Float64}(undef, length(_θ_est), length(_θ_est))

    # In case we fixate more parameters than there are chunk-size we might only want to evaluate ForwardDiff over a 
    # subset of chunks. To this end we here make sure "fixed" parameter are moved to the end of the parameter vector 
    # allowing us to take the chunks across the first parameters 
    __iθ_dynamicFix = [findfirst(x -> x == parameterFix, petabProblem.θ_indices.θ_dynamicNames) for parameterFix in parametersFix]
    _iθ_dynamicFix = __iθ_dynamicFix[findall(x -> !isnothing(x), __iθ_dynamicFix)]
    if !isempty(_iθ_dynamicFix) && length(_iθ_dynamicFix) ≥ 4
        k = 1
        for i in eachindex(petabProblem.θ_indices.θ_dynamicNames)
            i ∈ _iθ_dynamicFix && continue
            petabProblem.computeCost.petabODECache.θ_dynamicInputOrder[k] = i
            petabProblem.computeCost.petabODECache.θ_dynamicOutputOrder[i] = k
            k += 1
        end      
        petabProblem.computeCost.petabODECache.nθ_dynamicEst[1] = length(petabProblem.computeCost.petabODECache.θ_dynamicInputOrder) - length(_iθ_dynamicFix)

        # Aviod  problems with autodiff=true for ODE solvers for computing the gradient 
        if typeof(petabProblem.odeSolverGradientOptions.solver) <: Rodas5P
            petabProblem.odeSolverGradientOptions.solver = Rodas5P(autodiff=false)
        elseif typeof(petabProblem.odeSolverGradientOptions.solver) <: Rodas5
            petabProblem.odeSolverGradientOptions.solver = Rodas5(autodiff=false)
        elseif typeof(petabProblem.odeSolverGradientOptions.solver) <: Rodas4
            petabProblem.odeSolverGradientOptions.solver = Rodas4(autodiff=false)
        elseif typeof(petabProblem.odeSolverGradientOptions.solver) <: Rodas4P
            petabProblem.odeSolverGradientOptions.solver = Rodas4P(autodiff=false)
        elseif typeof(petabProblem.odeSolverGradientOptions.solver) <: Rosenbrock23
            petabProblem.odeSolverGradientOptions.solver = Rosenbrock23(autodiff=false)
        end
    else
        petabProblem.computeCost.petabODECache.θ_dynamicInputOrder .= 1:length(petabProblem.θ_indices.θ_dynamicNames)
        petabProblem.computeCost.petabODECache.θ_dynamicOutputOrder .= 1:length(petabProblem.θ_indices.θ_dynamicNames)
        petabProblem.computeCost.petabODECache.nθ_dynamicEst[1] = length(petabProblem.θ_indices.θ_dynamicNames)
    end

    # Setup mapping from θ_est to _θ_est
    iMap = [findfirst(x -> x == θ_estNames[i], petabProblem.θ_estNames) for i in eachindex(θ_estNames)]

    _computeCost = (θ_est) ->   begin
                                    __θ_est = convert.(eltype(θ_est), _θ_est)
                                    __θ_est[iParametersFix] .= parametersFixValues
                                    __θ_est[iMap] .= θ_est
                                    return petabProblem.computeCost(__θ_est)
                                end
    _computeSimulatedValues = (θ_est; asArray=false) -> begin
        __θ_est = convert.(eltype(θ_est), _θ_est)
        __θ_est[iParametersFix] .= parametersFixValues
        __θ_est[iMap] .= θ_est
        return petabProblem.computeSimulatedValues(__θ_est)
    end
    _computeChi2 = (θ_est) -> begin
        __θ_est = convert.(eltype(θ_est), _θ_est)
        __θ_est[iParametersFix] .= parametersFixValues
        __θ_est[iMap] .= θ_est
        return petabProblem.computeChi2(__θ_est)
    end
    _computeResiduals = (θ_est) -> begin
        __θ_est = convert.(eltype(θ_est), _θ_est)
        __θ_est[iParametersFix] .= parametersFixValues
        __θ_est[iMap] .= θ_est
        return petabProblem.computeResiduals(__θ_est)
    end

    _computeGradient! = (gradient, θ_est) ->    begin
                                                    __θ_est = convert.(eltype(θ_est), _θ_est)
                                                    __θ_est[iParametersFix] .= parametersFixValues
                                                    __θ_est[iMap] .= θ_est
                                                    if (petabProblem.gradientMethod === :ForwardDiff || petabProblem.gradientMethod === :ForwardEquations) && petabProblem.splitOverConditions == false
                                                        petabProblem.computeGradient!(_gradient, __θ_est; isRemade=true)
                                                    else
                                                        petabProblem.computeGradient!(_gradient, __θ_est)                                                    
                                                    end
                                                    gradient .= _gradient[iMap]
                                                end
    _computeGradient = (θ) -> begin
        gradient = zeros(Float64, length(θ))
        _computeGradient!(gradient, θ)
        return gradient
    end                                                                                     

    _computeHessian! = (hessian, θ_est) ->  begin 
                                                __θ_est = convert.(eltype(θ_est), _θ_est)
                                                __θ_est[iParametersFix] .= parametersFixValues
                                                __θ_est[iMap] .= θ_est
                                                if (petabProblem.gradientMethod === :GaussNewton) && petabProblem.splitOverConditions == false
                                                    petabProblem.computeHessian!(_hessian, __θ_est; isRemade=true)
                                                else
                                                    petabProblem.computeHessian!(_hessian, __θ_est)
                                                end
                                                # Can use double index with first and second 
                                                @inbounds for (i1, i2) in pairs(iMap)
                                                    for (j1, j2) in pairs(iMap)
                                                        hessian[i1, j1] = _hessian[i2, j2]
                                                    end
                                                end
                                            end
    _computeHessian = (θ) -> begin
        hessian = zeros(Float64, length(θ), length(θ))
        _computeHessian!(hessian, θ)
        return hessian
    end                                                                                                                                 

    _petabProblem = PEtabODEProblem(_computeCost,
                                    _computeChi2,
                                    _computeGradient!,
                                    _computeGradient,
                                    _computeHessian!,
                                    _computeHessian,
                                    _computeSimulatedValues, 
                                    _computeResiduals,
                                    petabProblem.costMethod,
                                    petabProblem.gradientMethod, 
                                    petabProblem.hessianMethod, 
                                    Int64(length(θ_estNames)),
                                    θ_estNames,
                                    θ_nominal,
                                    θ_nominalT,
                                    lowerBounds,
                                    upperBounds,
                                    petabProblem.pathCube,
                                    petabProblem.petabModel, 
                                    petabProblem.odeSolverOptions, 
                                    petabProblem.odeSolverGradientOptions, 
                                    petabProblem.ssSolverOptions, 
                                    petabProblem.ssSolverGradientOptions, 
                                    petabProblem.θ_indices,
                                    petabProblem.simulationInfo, 
                                    petabProblem.splitOverConditions)
    return _petabProblem
end