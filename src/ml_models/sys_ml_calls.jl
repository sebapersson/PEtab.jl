const SystemTrace = Union{ModelSystem, ModelingToolkitBase.System}

function _get_ml_model_for_ps(sys::ReactionSystem, ml_ps_id)
    ode_sys = _get_system(sys)
    return _get_ml_model_for_ps(ode_sys, ml_ps_id)
end
function _get_ml_model_for_ps(sys::SystemTrace, ml_ps_id)
    NNs = _ml_models_for_ps(sys, ml_ps_id)
    if isempty(NNs)
        throw(PEtabInputError("No neural network found using parameter $(ml_ps_id)."))
    end
    if length(NNs) > 1
        throw(PEtabInputError("Multiple neural networks found using parameter $(ml_ps_id): \
            $(NNs)."))
    end
    return only(NNs)
end

function _ml_models_for_ps(sys::SystemTrace, ml_ps_id)
    models = []
    for nn in _ml_callables(sys)
        uses_ps = any(equations(sys)) do eq
            _has_ml_call(eq.rhs, nn, ml_ps_id)
        end
        uses_ps && push!(models, nn)
    end
    return unique(models)
end

function _has_ml_call(expr, NN, ml_ps_id)::Bool
    return Symbolics.hasnode(expr) do x
        x = Symbolics.unwrap(x)

        Symbolics.iscall(x) || return false
        _same_symbolic(Symbolics.operation(x), NN) || return false

        args = Symbolics.arguments(x)
        isempty(args) && return false

        return _same_symbolic(args[end], ml_ps_id)
    end
end

_same_symbolic(a, b) = isequal(a, b) || isequal(Symbolics.unwrap(a), Symbolics.unwrap(b))

function _ml_callables(sys::SystemTrace)
    f_filter = (x) -> _is_neural_network_mtk(x, sys)
    return filter(f_filter, parameters(sys))
end

# This function retrieves "full" MLCalls. I.e. it finds ML function occurrences as `U(X1, X2, ...; θ)`
# And extracts a tuple (U, [X1, X2, ...], θ) for each occurrence.
function _get_full_ml_calls(petab_uprob::PEtab.PEtabODEProblem)
    sys = _get_system(petab_uprob.model_info.model.sys)
    return _get_full_ml_calls(sys)
end

function _get_full_ml_calls(sys::ModelingToolkitBase.AbstractSystem)
    full_ml_calls = Tuple[]
    for eq in ModelingToolkitBase.equations(sys)
        _collect_full_ml_calls!(full_ml_calls, eq.lhs, sys)
        _collect_full_ml_calls!(full_ml_calls, eq.rhs, sys)
    end
    return full_ml_calls
end

function _collect_full_ml_calls!(full_ml_calls, expr, sys)
    x = Symbolics.unwrap(expr)
    Symbolics.iscall(x) || return full_ml_calls

    args = collect(Symbolics.arguments(x))
    if !isempty(args) && (Symbolics.operation(x) isa Symbolics.SymbolicT) &&
            _is_neural_network_mtk(Symbolics.operation(x), sys)
        ps_sym = args[end]
        if _is_neural_network_mtk_ps(ps_sym, sys)
            nn_sym = try
                PEtab._get_ml_model_for_ps(sys, ps_sym)
            catch
                nothing
            end

            if !isnothing(nn_sym) && PEtab._same_symbolic(Symbolics.operation(x), nn_sym)
                inputs = _normalise_nn_args(args[1:(end - 1)])
                push!(full_ml_calls, (nn_sym, inputs, ps_sym))
            end
        end
    end

    for arg in args
        _collect_full_ml_calls!(full_ml_calls, arg, sys)
    end
    return full_ml_calls

end

# Takes into account that Neural network calls can occur like `U(X1, X2, ...; θ)` or `U([X1, X2, ...]; θ)`
function _normalise_nn_args(args)
    vars = Set{Symbolics.SymbolicT}()
    SymbolicUtils.search_variables!(vars, args)
    return collect(vars)
end


# Groups full ML calls by signature (NN model + NN parameterisation).
# Input format: vector of tuples `(nn_sym, input_args, ps_sym)` as returned by `_get_full_ml_calls`.
# Output format: vector of tuples `(nn_sym, ps_sym, all_input_args)` where
# `all_input_args` is a vector containing all observed input vectors for that signature.
# Accounts for one fitted function can occur with different input arguments (e.g. a repressilator
# where the same repressive function is fitted with inputs X, Y, and Z).
function _group_full_ml_calls_by_signature(full_ml_calls::AbstractVector)
    grouped = Tuple[]
    idx_by_sig = Dict{Tuple{Any, Any}, Int}()

    for call in full_ml_calls
        length(call) == 3 || throw(
            ArgumentError(
                "Each ML call must be a 3-tuple: (nn_sym, input_args, ps_sym). Got $(call)."
            )
        )
        nn_sym, input_args, ps_sym = call
        input_args isa AbstractVector || throw(
            ArgumentError(
                "The second tuple entry (input_args) must be a vector. Got $(typeof(input_args))."
            )
        )

        sig = (nn_sym, ps_sym)
        if haskey(idx_by_sig, sig)
            push!(grouped[idx_by_sig[sig]][3], input_args)
        else
            push!(grouped, (nn_sym, ps_sym, [input_args]))
            idx_by_sig[sig] = length(grouped)
        end
    end

    return grouped
end

"""
    get_fitted_functions(res::PEtab.EstimationResult, prob::PEtabODEProblem)

Returns a vector of functions, where each function is a function fitted as a machine learning
model. Each function is produced by the signature of a ML model (neural network) + parametrisation.

Notes:
- The functions output vectors of values. For functions R^n --> R^1, this means that they return
vectors of length 1, not scalars.
- Typically, neural networks and parametrisations are coupled, but this account for if the same
parametrisation is provided to different neural networks, or reverse (neither is recommended).
"""
function get_fitted_functions(res::PEtab.EstimationResult, prob::PEtabODEProblem)
    oprob, _ = get_odeproblem(res, prob)
    full_ml_calls = _get_full_ml_calls(prob)
    grouped_calls = _group_full_ml_calls_by_signature(full_ml_calls)

    fitted_functions = Function[]
    for (nn_sym, ps_sym, _) in grouped_calls
        fitted_nn = oprob.ps[nn_sym]
        fitted_ps = oprob.ps[ps_sym]

        # Accept either f(x1, x2, ...) or f([x1, x2, ...]) input style.
        f = function (args...)
            nn_args = _normalise_nn_args(collect(args))
            return fitted_nn(nn_args, fitted_ps)
        end
        push!(fitted_functions, f)
    end

    return fitted_functions
end
