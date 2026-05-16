const SystemTrace = Union{ModelSystem, ModelingToolkitBase.System}

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
    Symbolics.hasnode(expr) do x
        x = Symbolics.unwrap(x)

        Symbolics.iscall(x) || return false
        _same_symbolic(operation(x), NN) || return false

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
