function sol_compare_to(cbset, tstops)
    a0, b0, k1, k2 = [1.0, 0.0, 0.8, 0.6]
    _f! = function(du, u, p, t)
        k1, k2 = p
        A, B = u
        du[1] = -k1*A + k2*B
        du[2] = k1*A - k2*B
    end

    prob = ODEProblem(_f!, [a0, b0], (0.0, 10.0), [k1, k2])
    tstops = isnothing(tstops) ? Float64[] : tstops
    return solve(prob, Rodas5P(), abstol=1e-8, reltol=1e-8, tstops=tstops, callback=cbset)
end

function test_callbacks(events, cbset, tstops, float_tspan::Bool)
    # Measurement data
    measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                            time=[0, 10.0],
                            measurement=[0.7, 0.1],
                            noise_parameters=0.5)

    # PEtab-parameter to "estimate"
    parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                  PEtabParameter(:b0, value=0.0, scale=:lin),
                  PEtabParameter(:k1, value=0.8, scale=:lin),
                  PEtabParameter(:k2, value=0.6, scale=:lin, estimate=false)]

    # Observable equation
    @unpack A = rn
    observables = Dict("obs_a" => PEtabObservable(A, 0.5))

    # Controll case without any callback
    model = PEtabModel(rn, observables, measurements, parameters, events=events)
    petab_problem = PEtabODEProblem(model, verbose=false)
    p0 = petab_problem.xnominal_transformed
    sol_petab = get_odesol(p0, petab_problem)
    sol_compare = sol_compare_to(cbset, tstops)

    @test all(reduce(vcat, (sol_petab(sol_petab.t) .-= sol_compare(sol_petab.t))) .< 1e-4)
    @test model.float_tspan == float_tspan
end

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

@testset "PEtabEvent" begin
    test_callbacks(nothing, nothing, nothing, true)

    # At t == 5 A += 5
    @unpack A = rn
    event = PEtabEvent(5.0, A + 5, A)
    _condition1(u, t, integrator) = t == 5
    _affect1!(integrator) = integrator.u[1] += 5
    cbset = DiscreteCallback(_condition1, _affect1!)
    test_callbacks(event, cbset, [5.0], true)

    # At t == 5 A = 5
    @unpack A = rn
    event = PEtabEvent(5.0, 5, A)
    _condition2(u, t, integrator) = t == 5
    _affect2!(integrator) = integrator.u[1] = 5
    cbset = DiscreteCallback(_condition2, _affect2!)
    test_callbacks(event, cbset, [5.0], true)

    # When A < 0.8 add A = 1
    @unpack A = rn
    event = PEtabEvent(A < 0.8, 1.0, A)
    _condition3(u, t, integrator) = u[1] - 0.8
    _affect3!(integrator) = integrator.u[1] = 1.0
    cbset = ContinuousCallback(_condition3, _affect3!)
    test_callbacks(event, cbset, nothing, true)

    # When A > 0.8 add A = 1, nothing should happen as A comes from the wrong direction
    @unpack A = rn
    event = PEtabEvent(A > 0.8, 1.0, A)
    _condition4(u, t, integrator) = u[1] - 0.8
    _affect4!(integrator) = integrator.u[1] = 1.0
    cbset = ContinuousCallback(_condition4, _affect4!, affect_neg! = nothing)
    test_callbacks(event, cbset, nothing, true)

    # When A > 0.8 add A = 2.0, when B > 0.3 add B -> 5 (should trigger both)
    @unpack A, B = rn
    events = [PEtabEvent(A > 0.8, 2.0, A), PEtabEvent(B > 0.3, 5.0, B)]
    condition5(u, t, integrator) = u[1] - 0.8
    affect5!(integrator) = integrator.u[1] = 2.0
    condition6(u, t, integrator) = u[2] - 0.3
    affect6!(integrator) = integrator.u[2] = 5.0
    cbset = CallbackSet(ContinuousCallback(condition5, affect5!, affect_neg! = nothing),
                        ContinuousCallback(condition6, affect6!))
    test_callbacks(event, cbset, nothing, true)

    # When A == 0.8 add A = 2.0
    @unpack A = rn
    event = PEtabEvent(A == 0.8, 2.0, A)
    _condition7(u, t, integrator) = u[1] - 0.8
    _affect7!(integrator) = integrator.u[1] = 2.0
    cbset = ContinuousCallback(_condition7, _affect7!)
    test_callbacks(event, cbset, nothing, true)

    # When t == 4.0 change k2 -> 1.0
    @parameters t
    event = PEtabEvent(t == 4.0, 1.0, :k2)
    _condition8(u, t, integrator) = t == 4.0
    _affect8!(integrator) = integrator.p[2] = 1.0
    cbset = DiscreteCallback(_condition8, _affect8!)
    test_callbacks(event, cbset, [4.0], true)

    # When t == k2 change A -> 1.0
    @parameters t
    @unpack k2 = rn
    event = PEtabEvent(t == k2, 1.0, :A)
    _condition9(u, t, integrator) = t == integrator.p[2]
    _affect9!(integrator) = integrator.u[1] = 1.0
    cbset = DiscreteCallback(_condition9, _affect9!)
    test_callbacks(event, cbset, [0.6], true)

    # When t == k1 change A -> 1.0
    @parameters t
    event = PEtabEvent(:k1, 1.0, :A)
    _condition10(u, t, integrator) = t == integrator.p[1]
    _affect10!(integrator) = integrator.u[1] = 1.0
    cbset = DiscreteCallback(_condition10, _affect10!)
    test_callbacks(event, cbset, [0.8], false)

    # When t == k1 change A -> 1.0, B -> B + 2 (multiple targets and affects)
    @parameters t
    @unpack B = rn
    event = PEtabEvent(:k1, [1.0, B + 2], [:A, B])
    _condition11(u, t, integrator) = t == integrator.p[1]
    function _affect11!(integrator)
         integrator.u[1] = 1.0
         integrator.u[2] += 2.0
    end
    cbset = DiscreteCallback(_condition11, _affect11!)
    test_callbacks(event, cbset, [0.8], false)

    # When t = 1.5 change A -> 3.0, and when B == 1 -> B -> 3.0 (mix events)
    @unpack B = rn
    events = [PEtabEvent(:k1, 3.0, :A), PEtabEvent(B == 1.0, 3.0, B)]
    condition12(u, t, integrator) = t == integrator.p[1]
    affect12!(integrator) = integrator.u[1] = 3.0
    condition13(u, t, integrator) = u[2] - 1.0
    affect13!(integrator) = integrator.u[2] = 3.0
    cbset = CallbackSet(DiscreteCallback(condition12, affect12!),
                        ContinuousCallback(condition13, affect13!))
    test_callbacks(events, cbset, [0.8], false)

    measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                            time=[0, 10.0],
                            measurement=[0.7, 0.1],
                            noise_parameters=0.5)
    parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                        PEtabParameter(:b0, value=0.0, scale=:lin),
                        PEtabParameter(:k1, value=0.8, scale=:lin),
                        PEtabParameter(:k2, value=0.6, scale=:lin, estimate=false)]
    @unpack A = rn
    observables = Dict("obs_a" => PEtabObservable(A, 0.5))

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(0.5, 5.0, :C) # Target does not exist
        model = PEtabModel(rn, observables, measurements, parameters, verbose=false, events=event)
    end

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(:B, 5.0, :A) # Trigger cannot not only be a state
        PEtabModel(rn, observables, measurements, parameters, verbose=false, events=event)
    end

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(:k1, [1.0, 2.0], :A)
        PEtabModel(rn, observables, measurements, parameters, verbose=false, events=event)
    end

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(:k1, [1.0], [:A, :B])
        PEtabModel(rn, observables, measurements, parameters, verbose=false, events=event)
    end

    @test_throws PEtab.PEtabFormatError begin
        event = PEtabEvent(A + 3, 5.0, :A) # Trigger cannot not only be a state
        PEtabModel(rn, observables, measurements, parameters, verbose=false, events=event)
    end
end
