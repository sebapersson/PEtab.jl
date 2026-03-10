#=
    Event functionality added with PEtab v2. In particular:

    1. Events only applied to a subset of conditions
    2. Measurement-point at event time
=#

rn = @reaction_network begin
    @species begin
        A(t) = 5.0
        B(t) = 1.0
    end
    (k1, k2), A <--> B
end

_condition_ref1(u, t, integrator) = t == 7
_affect_ref1!(integrator) = integrator.u[1] += 5
cb_ref = DiscreteCallback(_condition_ref1, _affect_ref1!, save_positions = (false, false))
ode_prob_ref = ODEProblem(rn, [:A => 5.0, :B => 1.0], (0.0, 10.0), [:k1 => 0.8, :k2 => 0.6])

observables = PEtabObservable(:obs_a, "A", 1.0)
parameters = [
    PEtabParameter(:k1, value = 0.8, scale = :lin),
    PEtabParameter(:k2, value = 0.6, scale = :lin),
]
simulation_conditions = [PEtabCondition(:e1), PEtabCondition(:e2)]
measurements = DataFrame(
    simulation_id = ["e1", "e1", "e2", "e2"],
    obs_id = ["obs_a", "obs_a", "obs_a", "obs_a"],
    time = [5.0, 10.0, 5.0, 10.0],
    measurement = [2.0, 2.5, 2.1, 3.2]
)

# PEtab event only for simulation condition e1
sol_e1 = solve(
    ode_prob_ref, Rodas5P(), abstol = 1.0e-8, reltol = 1.0e-8, callback = cb_ref,
    tstops = [7.0], saveat = [5.0, 10.0]
)
sol_e2 = solve(ode_prob_ref, Rodas5P(), abstol = 1.0e-8, reltol = 1.0e-8, saveat = [5.0, 10.0])
nllh_ref = sum(0.5 .* (sol_e1[:A] - measurements.measurement[1:2]) .^ 2 .+ 0.5log(2π)) +
    sum(0.5 .* (sol_e2[:A] - measurements.measurement[3:4]) .^ 2 .+ 0.5log(2π))

event_e1 = PEtabEvent(7.0, "A" => "A + 5"; condition_ids = [:e1])
petab_model = PEtabModel(
    rn, observables, measurements, parameters; verbose = true,
    simulation_conditions = simulation_conditions, events = event_e1
)
petab_prob = PEtabODEProblem(petab_model)
nllh = petab_prob.nllh(get_x(petab_prob))

@test !isempty(petab_model.callbacks[:e1])
@test isempty(petab_model.callbacks[:e2])
@test nllh ≈ nllh_ref atol = 1.0e-3

# PEtab event applied to both simulation conditions
nllh_ref = sum(0.5 .* (sol_e1[:A] - measurements.measurement[1:2]) .^ 2 .+ 0.5log(2π)) +
    sum(0.5 .* (sol_e1[:A] - measurements.measurement[3:4]) .^ 2 .+ 0.5log(2π))

event_e1_e2 = PEtabEvent(7.0, "A" => "A + 5")
petab_model = PEtabModel(
    rn, observables, measurements, parameters;
    simulation_conditions = simulation_conditions, events = event_e1_e2
)
petab_prob = PEtabODEProblem(petab_model)
nllh = petab_prob.nllh(get_x(petab_prob))

@test !isempty(petab_model.callbacks[:e1])
@test !isempty(petab_model.callbacks[:e2])
@test length(petab_model.callbacks[:e1].discrete_callbacks) == 1
@test length(petab_model.callbacks[:e2].discrete_callbacks) == 1
@test nllh ≈ nllh_ref atol = 1.0e-3

# Measurement point at the same time-point as a PEtab event. Relies on PEtab inferring
# the event time (thus first checking the trigger time function) for relevant conditions
# Reference solution
_condition_ref2(u, t, integrator) = t == 10
_affect_ref2!(integrator) = integrator.u[1] += 5
cb_ref = DiscreteCallback(_condition_ref2, _affect_ref2!, save_positions = (false, true))
ode_prob_ref = ODEProblem(rn, [:A => 5.0, :B => 1.0], (0.0, 10.0), [:k1 => 0.8, :k2 => 0.6])
sol_e1 = solve(
    ode_prob_ref, Rodas5P(), abstol = 1.0e-8, reltol = 1.0e-8, callback = cb_ref,
    tstops = [7.0], saveat = [5.0]
)
sol_e2 = solve(
    ode_prob_ref, Rodas5P(), abstol = 1.0e-8, reltol = 1.0e-8, saveat = [5.0, 10.0]
)
nllh_ref = sum(0.5 .* (sol_e1[:A] - measurements.measurement[1:2]) .^ 2 .+ 0.5log(2π)) +
    sum(0.5 .* (sol_e2[:A] - measurements.measurement[3:4]) .^ 2 .+ 0.5log(2π))

event1 = PEtabEvent("t == 7.0", "A" => "A + 5"; condition_ids = [:e1])
event2 = PEtabEvent("7.0 == t", "A" => "A + 5"; condition_ids = [:e1])
event3 = PEtabEvent(7.0, "A" => "A + 5"; condition_ids = [:e1])
@test PEtab._get_trigger_time(event1) == 7.0
@test PEtab._get_trigger_time(event2) == 7.0
@test PEtab._get_trigger_time(event3) == 7.0

event_e1 = PEtabEvent(10.0, "A" => "A + 5"; condition_ids = [:e1])
petab_model = PEtabModel(
    rn, observables, measurements, parameters;
    simulation_conditions = simulation_conditions, events = event_e1
)
petab_prob = PEtabODEProblem(petab_model)
nllh = petab_prob.nllh(get_x(petab_prob))

@test petab_model.petab_events[1].trigger_time == 10.0
@test petab_prob.model_info.simulation_info.tsaves[:e1] == [5.0]
@test petab_prob.model_info.simulation_info.tsaves[:e2] == [5.0, 10.0]
@test nllh ≈ nllh_ref atol = 1.0e-3
