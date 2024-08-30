function PEtab._get_grad_f(method::Val{:Adjoint}, probleminfo::PEtab.PEtabODEProblemInfo,
                           model_info::PEtab.ModelInfo)::Tuple{Function, Function}
    @unpack gradient_method, sensealg, sensealg_ss, cache = probleminfo
    @unpack simulation_info = model_info
    @unpack xdynamic = cache

    _nllh_not_solve = PEtab._get_nllh_not_solveode(probleminfo, model_info; grad_adjoint = true)
    _compute_gradient! = let pinfo = probleminfo, minfo = model_info, _nllh_not_solve = _nllh_not_solve
        (g, x) -> grad_adjoint!(g, x, _nllh_not_solve, pinfo, minfo; cids = [:all])
    end

    _compute_gradient = let _compute_gradient! = _compute_gradient!
        (x) -> begin
            gradient = zeros(Float64, length(x))
            _compute_gradient!(gradient, x)
            return gradient
        end
    end
    return _compute_gradient!, _compute_gradient
end

function PEtab._get_sensealg(sensealg, ::Val{:Adjoint})
    allowed_methods = [InterpolatingAdjoint, QuadratureAdjoint, GaussAdjoint]
    if !isnothing(sensealg)
        @assert sensealg isa AdjointAlg "For gradient method :Adjoint allowed sensealg " *
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
