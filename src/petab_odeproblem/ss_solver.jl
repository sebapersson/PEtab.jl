function SteadyStateSolver(ss_solver::SteadyStateSolver, oprob::ODEProblem, osolver::ODESolver)::SteadyStateSolver
    abstol = isnothing(ss_solver.abstol) ? osolver.abstol * 100 : ss_solver.abstol
    reltol = isnothing(ss_solver.reltol) ? osolver.reltol * 100 : ss_solver.reltol
    maxiters = isnothing(ss_solver.maxiters) ? osolver.maxiters : ss_solver.maxiters
    @unpack pseudoinverse, method, termination_check, rootfinding_alg = ss_solver
    if method === :Simulate
        if termination_check === :Newton
            newton = true
            jac = zeros(Float64, length(oprob.u0), length(oprob.u0))
        else
            newton = false
            jac = zeros(Float64, 0, 0)
        end
        condss = (u, t, integrator) -> condition_ss(u, t, integrator, abstol, reltol,
                                                    newton, oprob.f.jac, jac, pseudoinverse)
        callback_ss = DiscreteCallback(condss, affect_ss!, save_positions = (false, true))
    else
        callback_ss = nothing
    end
    return SteadyStateSolver(method, rootfinding_alg, termination_check, abstol, reltol,
                             maxiters, callback_ss, NonlinearProblem(oprob), pseudoinverse)
end

# Callback in case steady-state is found via  model simulation
function condition_ss(u, t, integrator, abstol::Float64, reltol::Float64,
                      newton::Bool, jacobian!::Function, jac::AbstractMatrix,
                      pseudoinverse::Bool)::Bool
    testval = first(get_tmp_cache(integrator))
    DiffEqBase.get_du!(testval, integrator)

    success_newton = true
    local Δu
    # Check Termination via a Newton-step. For this to work Jacobian should be invertiable
    if newton == true
        # Important all computatations are performed with Floats. TODO: Figure out how
        # to deal with dual numbers
        _u = SBMLImporter._to_float.(u)
        _p = SBMLImporter._to_float.(integrator.p)
        _t = SBMLImporter._to_float(t)
        jacobian!(jac, _u, _p, _t)
        try
            Δu = jac \ testval
            success_newton = true
        catch
            success_newton = false
        end
    end

    if newton == true && success_newton == false && pseudoinverse == true
        @warn "Jacobian non-invertible when solving for steady-state. " *
              "By user option uses pseduo instead (displays max 10 times)" maxlog=10
        Δu = pinv(jac) * testval
    elseif newton == true && success_newton == false
        @warn "Jacobian non-invertible when solving for steady-state. " *
              "By default uses wrms instead (displays max 10 times)" maxlog=10
    end

    nu = length(u)
    if newton && (success_newton == true || pseudoinverse == true)
        valcheck = sqrt(sum((Δu / (reltol * integrator.u .+ abstol)) .^ 2) / nu)
    else
        valcheck = sqrt(sum((testval ./ (reltol * integrator.u .+ abstol)) .^ 2) / nu)
    end
    return valcheck < 1.0
end

function affect_ss!(integrator)
    terminate!(integrator)
end
