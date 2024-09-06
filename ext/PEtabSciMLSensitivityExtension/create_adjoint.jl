function PEtab._get_grad(method::Val{:Adjoint}, probinfo::PEtab.PEtabODEProblemInfo,
                         model_info::PEtab.ModelInfo,
                         grad_prior::Function)::Tuple{Function, Function}
    @unpack gradient_method, sensealg, sensealg_ss, cache = probinfo
    @unpack simulation_info = model_info
    @unpack xdynamic = cache

    _nllh_not_solve = PEtab._get_nllh_not_solveode(probinfo, model_info;
                                                   grad_adjoint = true)
    _grad_nllh! = let pinfo = probinfo, minfo = model_info,
        _nllh_not_solve = _nllh_not_solve

        (g, x) -> grad_adjoint!(g, x, _nllh_not_solve, pinfo, minfo; cids = [:all])
    end

    _grad! = let _grad_nllh! = _grad_nllh!, grad_prior = grad_prior
        (g, x; prior = true) -> begin
            _grad_nllh!(g, x)
            if prior
                # nllh -> negative prior
                g .+= grad_prior(x) .* -1
            end
            return nothing
        end
    end
    _grad = let _grad! = _grad!
        (x; prior = true) -> begin
            gradient = zeros(Float64, length(x))
            _grad!(gradient, x; prior = prior)
            return gradient
        end
    end
    return _grad!, _grad
end

function PEtab._get_sensealg(sensealg, ::Val{:Adjoint})
    allowed_methods = [InterpolatingAdjoint, QuadratureAdjoint, GaussAdjoint]
    if !isnothing(sensealg)
        @assert sensealg isa AdjointAlg "For gradient method :Adjoint allowed sensealg "*
        "args $allowed_methods not $sensealg"
        return sensealg
    end
    return InterpolatingAdjoint(autojacvec = ReverseDiffVJP())
end
function PEtab._get_sensealg(sensealg::ForwardAlg, ::Val{:ForwardEquations})
    return sensealg
end

function PEtab._get_sensealg_ss(sensealg_ss, sensealg, model_info::PEtab.ModelInfo,
                                ::Val{:Adjoint})
    model_info.simulation_info.has_pre_equilibration == false && return nothing
    # Fast but numerically unstable method
    if sensealg_ss isa SteadyStateAdjoint
        throw(PEtabFormatError("Currently we do not support SteadyStateAdjoint. We are " *
                               "working on it"))
    end
    sensealg_ss_use = isnothing(sensealg_ss) ? sensealg : sensealg_ss
    # If sensealg_ss = GaussAdjoint as we do not actually have any observations during the
    # pre-eq simulations, there is no difference between using Guass and Interpolating
    # adjoint. Hence, to keep the size of the code-base smaller we use Gauss-adjoint
    if sensealg_ss_use isa GaussAdjoint
        sensealg_ss_use = InterpolatingAdjoint(autojacvec = sensealg_ss_use.autojacvec)
    end
    return sensealg_ss_use
end
