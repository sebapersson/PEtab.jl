using OrdinaryDiffEq
using Catalyst
using DataFrames
using PEtab
using Test


function get_ode_sol(θ::Vector{Float64},
                     petab_problem::PEtabODEProblem;
                     condition_id::Union{String, Symbol, Nothing}=nothing,
                     pre_eq_id::Union{String, Symbol, Nothing}=nothing)

    @unpack simulation_info, ode_solver = petab_problem
    if isnothing(condition_id)
        condition_id = simulation_info.simulation_condition_id[1]
    end

    u0, p = PEtab._get_fitted_parameters(θ, petab_problem, condition_id, pre_eq_id, false)
    tmax = petab_problem.simulation_info.tmax[condition_id]
    ode_problem = remake(petab_problem.ode_problem, p=p, u0=u0, tspan=(0.0, tmax))

    cbset = petab_problem.petab_model.model_callbacks
    tstops = petab_problem.petab_model.compute_tstops(u0, p)

    @unpack solver, abstol, reltol = ode_solver
    return solve(ode_problem, solver, abstol=abstol, reltol=reltol, callback=cbset, tstops=tstops)
end


function sol_compare_to(cbset, tstops)
    a0, b0, k1, k2 = [1.0, 0.0, 0.8, 0.6]
    _f! = function(du, u, p, t)
        k1, k2 = p
        A, B = u
        du[1] = -k1*A + k2*B
        du[2] = k1*A - k2*B
    end

    prob = ODEProblem(_f!, [a0, b0], (0.0, 10.0), [k1, k2])
    tstops = isnothing(tstops) ? Vector{Float64}(undef, 0) : tstops
    return solve(prob, Rodas5P(), abstol=1e-8, reltol=1e-8, tstops=tstops, callback=cbset)
end


function test_callbacks(events, cbset, tstops, convert_tspan::Bool)

    # Measurement data
    measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                            time=[0, 10.0],
                            measurement=[0.7, 0.1],
                            noise_parameters=0.5)

    # PEtab-parameter to "estimate"
    petab_parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                        PEtabParameter(:b0, value=0.0, scale=:lin),
                        PEtabParameter(:k1, value=0.8, scale=:lin),
                        PEtabParameter(:k2, value=0.6, scale=:lin, estimate=false)]

    # Observable equation
    @unpack A = rn
    observables = Dict("obs_a" => PEtabObservable(A, 0.5))

    # Controll case without any callback
    petab_model = PEtabModel(rn, observables, measurements,
                            petab_parameters, verbose=false, events=events)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)
    p0 = petab_problem.θ_nominalT
    sol_petab = get_ode_sol(p0, petab_problem)
    sol_compare = sol_compare_to(cbset, tstops)

    @test all(reduce(vcat, (sol_petab(sol_petab.t) .-= sol_compare(sol_petab.t))) .< 1e-4)
    @test petab_model.convert_tspan == convert_tspan
end


rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end


@testset "PEtabEvent" begin
    test_callbacks(nothing, nothing, nothing, false)

    # At t == 5 A += 5
    @unpack A = rn
    event = PEtabEvent(5.0, A + 5, A)
    _condition(u, t, integrator) = t == 5
    _affect!(integrator) = integrator.u[1] += 5
    cbset = DiscreteCallback(_condition, _affect!)
    test_callbacks(event, cbset, [5.0], false)

    # At t == 5 A = 5
    @unpack A = rn
    event = PEtabEvent(5.0, 5, A)
    _condition(u, t, integrator) = t == 5
    _affect!(integrator) = integrator.u[1] = 5
    cbset = DiscreteCallback(_condition, _affect!)
    test_callbacks(event, cbset, [5.0], false)

    # When A < 0.8 add A = 1
    @unpack A = rn
    event = PEtabEvent(A < 0.8, 1.0, A)
    _condition(u, t, integrator) = u[1] - 0.8
    _affect!(integrator) = integrator.u[1] = 1.0
    cbset = ContinuousCallback(_condition, _affect!)
    test_callbacks(event, cbset, nothing, false)

    # When A > 0.8 add A = 1, nothing should happen as A comes from the wrong direction
    @unpack A = rn
    event = PEtabEvent(A > 0.8, 1.0, A)
    _condition(u, t, integrator) = u[1] - 0.8
    _affect!(integrator) = integrator.u[1] = 1.0
    cbset = ContinuousCallback(_condition, _affect!, affect_neg! = nothing)
    test_callbacks(event, cbset, nothing, false)


    # When A > 0.8 add A = 2.0, when B > 0.3 add B -> 5 (should trigger both)
    @unpack A, B = rn
    events = [PEtabEvent(A > 0.8, 2.0, A), PEtabEvent(B > 0.3, 5.0, B)]
    condition1(u, t, integrator) = u[1] - 0.8
    affect1!(integrator) = integrator.u[1] = 2.0
    condition2(u, t, integrator) = u[2] - 0.3
    affect2!(integrator) = integrator.u[2] = 5.0
    cbset = CallbackSet(ContinuousCallback(condition1, affect1!, affect_neg! = nothing),
                        ContinuousCallback(condition2, affect2!))
    test_callbacks(event, cbset, nothing, false)

    # When A == 0.8 add A = 2.0
    @unpack A = rn
    event = PEtabEvent(A == 0.8, 2.0, A)
    _condition(u, t, integrator) = u[1] - 0.8
    _affect!(integrator) = integrator.u[1] = 2.0
    cbset = ContinuousCallback(_condition, _affect!)
    test_callbacks(event, cbset, nothing, false)

    # When t == 4.0 change k2 -> 1.0
    @parameters t
    event = PEtabEvent(t == 4.0, 1.0, :k2)
    _condition(u, t, integrator) = t == 4.0
    _affect!(integrator) = integrator.p[2] = 1.0
    cbset = DiscreteCallback(_condition, _affect!)
    test_callbacks(event, cbset, [4.0], false)

    # When t == k2 change A -> 1.0
    @parameters t
    @unpack k2 = rn
    event = PEtabEvent(t == k2, 1.0, :A)
    _condition(u, t, integrator) = t == integrator.p[2]
    _affect!(integrator) = integrator.u[1] = 1.0
    cbset = DiscreteCallback(_condition, _affect!)
    test_callbacks(event, cbset, [0.6], false)

    # When t == k1 change A -> 1.0
    @parameters t
    event = PEtabEvent(:k1, 1.0, :A)
    _condition(u, t, integrator) = t == integrator.p[1]
    _affect!(integrator) = integrator.u[1] = 1.0
    cbset = DiscreteCallback(_condition, _affect!)
    test_callbacks(event, cbset, [0.8], true)

    # When t = 1.5 change A -> 3.0, and when B == 1 -> B -> 3.0 (mix events)
    @unpack B = rn
    events = [PEtabEvent(:k1, 3.0, :A), PEtabEvent(B == 1.0, 3.0, B)]
    condition1(u, t, integrator) = t == integrator.p[1]
    affect1!(integrator) = integrator.u[1] = 3.0
    condition2(u, t, integrator) = u[2] - 1.0
    affect2!(integrator) = integrator.u[2] = 3.0
    cbset = CallbackSet(DiscreteCallback(condition1, affect1!),
                        ContinuousCallback(condition2, affect2!))
    test_callbacks(event, cbset, [0.8], true)

    measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                            time=[0, 10.0],
                            measurement=[0.7, 0.1],
                            noise_parameters=0.5)
    petab_parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                        PEtabParameter(:b0, value=0.0, scale=:lin),
                        PEtabParameter(:k1, value=0.8, scale=:lin),
                        PEtabParameter(:k2, value=0.6, scale=:lin, estimate=false)]
    @unpack A = rn
    observables = Dict("obs_a" => PEtabObservable(A, 0.5))

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(0.5, 5.0, :C) # Target does not exist
        petab_model = PEtabModel(rn, observables, measurements, petab_parameters, verbose=false, events=event)
    end

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(:B, 5.0, :A) # Trigger cannot not only be a state
        PEtabModel(rn, observables, measurements, petab_parameters, verbose=false, events=event)
    end

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(A + 3, 5.0, :A) # Trigger cannot not only be a stte
        PEtabModel(rn, observables, measurements, petab_parameters, verbose=false, events=event)
    end
end


