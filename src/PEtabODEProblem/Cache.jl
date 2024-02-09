function PEtabODEProblemCache(gradient_method::Symbol,
                              hessian_method::Union{Symbol, Nothing},
                              FIM_method::Symbol,
                              petab_model::PEtabModel,
                              sensealg,
                              measurement_info::MeasurementsInfo,
                              simulation_info::SimulationInfo,
                              θ_indices::ParameterIndices,
                              _chunksize)::PEtabODEProblemCache
    θ_dynamic = zeros(Float64, length(θ_indices.iθ_dynamic))
    θ_observable = zeros(Float64, length(θ_indices.iθ_observable))
    θ_sd = zeros(Float64, length(θ_indices.iθ_sd))
    θ_non_dynamic = zeros(Float64, length(θ_indices.iθ_non_dynamic))

    level_cache = 0
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        level_cache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        level_cache = 1
    else
        level_cache = 0
    end

    # This ensures that the chunksize is not to small when computing Hessians
    chunksize = length(θ_indices.θ_names) * 2 + length(θ_indices.θ_names)^2

    _θ_dynamicT = zeros(Float64, length(θ_indices.iθ_dynamic))
    _θ_observableT = zeros(Float64, length(θ_indices.iθ_observable))
    _θ_sdT = zeros(Float64, length(θ_indices.iθ_sd))
    _θ_non_dynamicT = zeros(Float64, length(θ_indices.iθ_non_dynamic))
    θ_dynamicT = DiffCache(_θ_dynamicT, chunksize, levels = level_cache)
    θ_observableT = DiffCache(_θ_observableT, chunksize, levels = level_cache)
    θ_sdT = DiffCache(_θ_sdT, chunksize, levels = level_cache)
    θ_non_dynamicT = DiffCache(_θ_non_dynamicT, chunksize, levels = level_cache)

    gradient_θ_dyanmic = zeros(Float64, length(θ_dynamic))
    gradient_θ_not_ode = zeros(Float64, length(θ_indices.iθ_not_ode))

    # For forward sensitivity equations and adjoint sensitivity analysis we need to
    # compute partial derivatives symbolically. Here the helping vectors are pre-allocated
    if gradient_method ∈ [:Adjoint, :ForwardEquations] || hessian_method == :GaussNewton ||
       FIM_method == :GaussNewton
        n_model_states = length(states(petab_model.system_mutated))
        n_model_parameters = length(parameters(petab_model.system_mutated))
        ∂h∂u = zeros(Float64, n_model_states)
        ∂σ∂u = zeros(Float64, n_model_states)
        ∂h∂p = zeros(Float64, n_model_parameters)
        ∂σ∂p = zeros(Float64, n_model_parameters)
        ∂G∂p = zeros(Float64, n_model_parameters)
        ∂G∂p_ = zeros(Float64, n_model_parameters)
        ∂G∂u = zeros(Float64, n_model_states)
        p = zeros(Float64, n_model_parameters)
        u = zeros(Float64, n_model_states)
    else
        ∂h∂u = zeros(Float64, 0)
        ∂σ∂u = zeros(Float64, 0)
        ∂h∂p = zeros(Float64, 0)
        ∂σ∂p = zeros(Float64, 0)
        ∂G∂p = zeros(Float64, 0)
        ∂G∂p_ = zeros(Float64, 0)
        ∂G∂u = zeros(Float64, 0)
        p = zeros(Float64, 0)
        u = zeros(Float64, 0)
    end

    # In case the sensitivites are computed via automatic differentitation we need to pre-allocate an
    # sensitivity matrix all experimental conditions (to efficiently levarage autodiff and handle scenarios are
    # pre-equlibrita model). Here we pre-allocate said matrix and the output matrix from the forward senstivity
    # code
    if (gradient_method === :ForwardEquations && sensealg === :ForwardDiff) ||
       hessian_method === :GaussNewton || FIM_method == :GaussNewton
        n_model_states = length(states(petab_model.system_mutated))
        n_timepoints_save = sum(length(simulation_info.time_observed[experimental_condition_id])
                                for experimental_condition_id in simulation_info.experimental_condition_id)
        S = zeros(Float64,
                  (n_timepoints_save * n_model_states, length(θ_indices.θ_dynamic_names)))
        sol_values = zeros(Float64, n_model_states, n_timepoints_save)
    else
        S = zeros(Float64, (0, 0))
        sol_values = zeros(Float64, (0, 0))
    end

    if hessian_method === :GaussNewton || FIM_method === :GaussNewton
        jacobian_gn = zeros(Float64, length(θ_indices.θ_names),
                            length(measurement_info.time))
        residuals_gn = zeros(Float64, length(measurement_info.time))
    else
        jacobian_gn = zeros(Float64, (0, 0))
        residuals_gn = zeros(Float64, 0)
    end

    if gradient_method === :ForwardEquations || hessian_method === :GaussNewton ||
       FIM_method === :GaussNewton
        _gradient = zeros(Float64, length(θ_indices.iθ_dynamic))
    else
        _gradient = zeros(Float64, 0)
    end

    if gradient_method === :Adjoint
        n_model_states = length(states(petab_model.system_mutated))
        n_model_parameters = length(parameters(petab_model.system_mutated))
        du = zeros(Float64, n_model_states)
        dp = zeros(Float64, n_model_parameters)
        _gradient_adjoint = zeros(Float64, n_model_parameters)
        S_t0 = zeros(Float64, (n_model_states, n_model_parameters))
    else
        du = zeros(Float64, 0)
        dp = zeros(Float64, 0)
        _gradient_adjoint = zeros(Float64, 0)
        S_t0 = zeros(Float64, (0, 0))
    end

    # Allocate arrays to track if θ_dynamic should be permuted prior and post gradient compuations. This feature
    # is used if PEtabODEProblem is remade (via remake) to compute the gradient of a problem with reduced number
    # of parameters where to run fewer chunks with ForwardDiff.jl we only run enough chunks to reach nθ_dynamic
    θ_dynamic_input_order::Vector{Int64} = collect(1:length(θ_dynamic))
    θ_dynamic_output_order::Vector{Int64} = collect(1:length(θ_dynamic))
    nθ_dynamic::Vector{Int64} = Int64[length(θ_dynamic)]

    petab_ODE_cache = PEtabODEProblemCache(θ_dynamic,
                                           θ_sd,
                                           θ_observable,
                                           θ_non_dynamic,
                                           θ_dynamicT,
                                           θ_sdT,
                                           θ_observableT,
                                           θ_non_dynamicT,
                                           gradient_θ_dyanmic,
                                           gradient_θ_not_ode,
                                           jacobian_gn,
                                           residuals_gn,
                                           _gradient,
                                           _gradient_adjoint,
                                           S_t0,
                                           ∂h∂u,
                                           ∂σ∂u,
                                           ∂h∂p,
                                           ∂σ∂p,
                                           ∂G∂p,
                                           ∂G∂p_,
                                           ∂G∂u,
                                           dp,
                                           du,
                                           p,
                                           u,
                                           S,
                                           sol_values,
                                           θ_dynamic_input_order,
                                           θ_dynamic_output_order,
                                           nθ_dynamic)

    return petab_ODE_cache
end

function PEtabODESolverCache(gradient_method::Symbol,
                             hessian_method::Union{Symbol, Nothing},
                             petab_model::PEtabModel,
                             simulation_info::SimulationInfo,
                             θ_indices::ParameterIndices,
                             _chunksize)::PEtabODESolverCache
    n_model_states = length(states(petab_model.system_mutated))
    n_model_parameters = length(parameters(petab_model.system_mutated))

    level_cache = 0
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        level_cache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        level_cache = 1
    else
        level_cache = 0
    end

    chunksize = length(θ_indices.θ_names) * 2 + length(θ_indices.θ_names)^2

    if simulation_info.has_pre_equilibration_condition_id == true
        conditions_simulate_over = unique(vcat(simulation_info.pre_equilibration_condition_id,
                                               simulation_info.experimental_condition_id))
    else
        conditions_simulate_over = unique(simulation_info.experimental_condition_id)
    end

    _p_ode_problem_cache = Tuple(DiffCache(zeros(Float64, n_model_parameters), chunksize,
                                           levels = level_cache)
                                 for i in eachindex(conditions_simulate_over))
    _u0_cache = Tuple(DiffCache(zeros(Float64, n_model_states), chunksize,
                                levels = level_cache)
                      for i in eachindex(conditions_simulate_over))
    p_ode_problem_cache::Dict = Dict([(conditions_simulate_over[i],
                                       _p_ode_problem_cache[i])
                                      for i in eachindex(_p_ode_problem_cache)])
    u0_cache::Dict = Dict([(conditions_simulate_over[i], _u0_cache[i])
                           for i in eachindex(_u0_cache)])

    return PEtabODESolverCache(p_ode_problem_cache, u0_cache)
end
