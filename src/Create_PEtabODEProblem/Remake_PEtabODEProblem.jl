"""
    remake_PEtab_problem(petab_problem::PEtabODEProblem, parameters_change::Dict) :: PEtabODEProblem

Fixate model parameters for a given PEtabODEProblem without recompiling the problem.

This function allows you to modify parameters without the need to recompile the underlying code, resulting in reduced
latency. To fixate the parameter k1, you can use `parameters_change=Dict(:k1 => 1.0)`.

If model derivatives are computed using ForwardDiff.jl with a chunk-size of N, the new PEtabODEProblem will only
evaluate the necessary number of chunks of size N to compute the full gradient for the remade problem.
"""
function remake_PEtab_problem(petab_problem::PEtabODEProblem, parameters_change::Dict)::PEtabODEProblem

    # Only keep which parameters should be fixed
    for key in keys(parameters_change)
        if parameters_change[key] == "estimate"
            key ∉ petab_problem.θ_names && @error "When remaking an PEtab problem we cannot set new parameters in addition to those in the PEtab-file to be estimated"
            delete!(parameters_change, key)
        end
    end

    parametersFix = collect(keys(parameters_change))
    iParametersFix = [findfirst(x -> x == parameterFix, petab_problem.θ_names) for parameterFix in parametersFix]
    parametersFixValues = Vector{Float64}(undef, length(parametersFix))
    # Ensure we fixate parameter values on the correct scale
    for i in eachindex(iParametersFix)
        transform = petab_problem.compute_cost.parameterInfo.parameterScale[findfirst(x -> x == parametersFix[i], petab_problem.compute_cost.parameterInfo.parameterId)]
        if transform === :lin
            parametersFixValues[i] = parameters_change[parametersFix[i]]
        elseif transform === :log
            parametersFixValues[i] = log(parameters_change[parametersFix[i]])
        elseif transform === :log10
            parametersFixValues[i] = log10(parameters_change[parametersFix[i]])
        end
    end

    # Setup parameters for new problem of lower dimension
    iUse = findall(x -> x ∉ parametersFix, petab_problem.θ_names)
    lower_bounds = petab_problem.lower_bounds[iUse]
    upper_bounds = petab_problem.upper_bounds[iUse]
    θ_names = petab_problem.θ_names[iUse]
    θ_nominal = petab_problem.θ_nominal[iUse]
    θ_nominalT = petab_problem.θ_nominalT[iUse]

    # Gradient place-holders for the underlaying functions
    _θ_est::Vector{Float64} = similar(petab_problem.lower_bounds)
    _gradient::Vector{Float64} = similar(_θ_est)
    _hessian::Matrix{Float64} = Matrix{Float64}(undef, length(_θ_est), length(_θ_est))

    # In case we fixate more parameters than there are chunk-size we might only want to evaluate ForwardDiff over a
    # subset of chunks. To this end we here make sure "fixed" parameter are moved to the end of the parameter vector
    # allowing us to take the chunks across the first parameters
    __iθ_dynamicFix = [findfirst(x -> x == parameterFix, petab_problem.θ_indices.θ_dynamicNames) for parameterFix in parametersFix]
    _iθ_dynamicFix = __iθ_dynamicFix[findall(x -> !isnothing(x), __iθ_dynamicFix)]
    if !isempty(_iθ_dynamicFix)
        k = 1
        # Make sure the parameter which are to be "estimated" end up in the from
        # of the parameter vector when running ForwardDiff
        for i in eachindex(petab_problem.θ_indices.θ_dynamicNames)
            if i ∈ _iθ_dynamicFix
                continue
            end
            petab_problem.compute_cost.petabODECache.θ_dynamicInputOrder[k] = i
            petab_problem.compute_cost.petabODECache.θ_dynamicOutputOrder[i] = k
            k += 1
        end
        # Make sure the parameter which are fixated ends up in the end of the parameter
        # vector when running ForwardDiff
        for i in eachindex(petab_problem.θ_indices.θ_dynamicNames)
            if i ∉ _iθ_dynamicFix
                continue
            end
            petab_problem.compute_cost.petabODECache.θ_dynamicInputOrder[k] = i
            petab_problem.compute_cost.petabODECache.θ_dynamicOutputOrder[i] = k
            k += 1
        end
        petab_problem.compute_cost.petabODECache.nθ_dynamicEst[1] = length(petab_problem.θ_indices.θ_dynamicNames) - length(_iθ_dynamicFix)

        # Aviod  problems with autodiff=true for ODE solvers for computing the gradient
        if typeof(petab_problem.ode_solver_gradient.solver) <: Rodas5P
            petab_problem.ode_solver_gradient.solver = Rodas5P(autodiff=false)
        elseif typeof(petab_problem.ode_solver_gradient.solver) <: Rodas5
            petab_problem.ode_solver_gradient.solver = Rodas5(autodiff=false)
        elseif typeof(petab_problem.ode_solver_gradient.solver) <: Rodas4
            petab_problem.ode_solver_gradient.solver = Rodas4(autodiff=false)
        elseif typeof(petab_problem.ode_solver_gradient.solver) <: Rodas4P
            petab_problem.ode_solver_gradient.solver = Rodas4P(autodiff=false)
        elseif typeof(petab_problem.ode_solver_gradient.solver) <: Rosenbrock23
            petab_problem.ode_solver_gradient.solver = Rosenbrock23(autodiff=false)
        end
    else
        petab_problem.compute_cost.petabODECache.θ_dynamicInputOrder .= 1:length(petab_problem.θ_indices.θ_dynamicNames)
        petab_problem.compute_cost.petabODECache.θ_dynamicOutputOrder .= 1:length(petab_problem.θ_indices.θ_dynamicNames)
        petab_problem.compute_cost.petabODECache.nθ_dynamicEst[1] = length(petab_problem.θ_indices.θ_dynamicNames)
    end

    # Setup mapping from θ_est to _θ_est
    iMap = [findfirst(x -> x == θ_names[i], petab_problem.θ_names) for i in eachindex(θ_names)]

    _compute_cost = (θ_est) ->   begin
                                    __θ_est = convert.(eltype(θ_est), _θ_est)
                                    __θ_est[iParametersFix] .= parametersFixValues
                                    __θ_est[iMap] .= θ_est
                                    return petab_problem.compute_cost(__θ_est)
                                end
    _compute_simulated_values = (θ_est; asArray=false) -> begin
        __θ_est = convert.(eltype(θ_est), _θ_est)
        __θ_est[iParametersFix] .= parametersFixValues
        __θ_est[iMap] .= θ_est
        return petab_problem.compute_simulated_values(__θ_est)
    end
    _compute_chi2 = (θ_est) -> begin
        __θ_est = convert.(eltype(θ_est), _θ_est)
        __θ_est[iParametersFix] .= parametersFixValues
        __θ_est[iMap] .= θ_est
        return petab_problem.compute_chi2(__θ_est)
    end
    _compute_residuals = (θ_est) -> begin
        __θ_est = convert.(eltype(θ_est), _θ_est)
        __θ_est[iParametersFix] .= parametersFixValues
        __θ_est[iMap] .= θ_est
        return petab_problem.compute_residuals(__θ_est)
    end

    _compute_gradient! = (gradient, θ_est) ->    begin
                                                    __θ_est = convert.(eltype(θ_est), _θ_est)
                                                    __θ_est[iParametersFix] .= parametersFixValues
                                                    __θ_est[iMap] .= θ_est
                                                    if (petab_problem.gradient_method === :ForwardDiff || petab_problem.gradient_method === :ForwardEquations) && petab_problem.split_over_conditions == false
                                                        petab_problem.compute_gradient!(_gradient, __θ_est; isRemade=true)
                                                    else
                                                        petab_problem.compute_gradient!(_gradient, __θ_est)
                                                    end
                                                    gradient .= _gradient[iMap]
                                                end
    _compute_gradient = (θ) -> begin
        gradient = zeros(Float64, length(θ))
        _compute_gradient!(gradient, θ)
        return gradient
    end

    _compute_hessian! = (hessian, θ_est) ->  begin
                                                __θ_est = convert.(eltype(θ_est), _θ_est)
                                                __θ_est[iParametersFix] .= parametersFixValues
                                                __θ_est[iMap] .= θ_est
                                                if (petab_problem.gradient_method === :GaussNewton) && petab_problem.split_over_conditions == false
                                                    petab_problem.compute_hessian!(_hessian, __θ_est; isRemade=true)
                                                else
                                                    petab_problem.compute_hessian!(_hessian, __θ_est)
                                                end
                                                # Can use double index with first and second
                                                @inbounds for (i1, i2) in pairs(iMap)
                                                    for (j1, j2) in pairs(iMap)
                                                        hessian[i1, j1] = _hessian[i2, j2]
                                                    end
                                                end
                                            end
    _compute_hessian = (θ) -> begin
        hessian = zeros(Float64, length(θ), length(θ))
        _compute_hessian!(hessian, θ)
        return hessian
    end

    _petab_problem = PEtabODEProblem(_compute_cost,
                                    _compute_chi2,
                                    _compute_gradient!,
                                    _compute_gradient,
                                    _compute_hessian!,
                                    _compute_hessian,
                                    _compute_simulated_values,
                                    _compute_residuals,
                                    petab_problem.cost_method,
                                    petab_problem.gradient_method,
                                    petab_problem.hessian_method,
                                    Int64(length(θ_names)),
                                    θ_names,
                                    θ_nominal,
                                    θ_nominalT,
                                    lower_bounds,
                                    upper_bounds,
                                    petab_problem.pathCube,
                                    petab_problem.petab_model,
                                    petab_problem.ode_solver,
                                    petab_problem.ode_solver_gradient,
                                    petab_problem.ss_solver,
                                    petab_problem.ss_solver_gradient,
                                    petab_problem.θ_indices,
                                    petab_problem.simulation_info,
                                    petab_problem.split_over_conditions)
    return _petab_problem
end