function getCallbacks_Smith_BMCSystBiol2013()

	function condition_insulin_restimulation_end(u, t, integrator)
		t == 2895
	end

	function affect_insulin_restimulation_end!(integrator)
		integrator.u[103] = 0.0
	end

	cb_insulin_restimulation_end = DiscreteCallback(condition_insulin_restimulation_end, affect_insulin_restimulation_end!, save_positions=(false, false))


	function condition_insulin_stimulation_end(u, t, integrator)
		t == integrator.p[68]
	end

	function affect_insulin_stimulation_end!(integrator)
		integrator.u[103] = 0.0
	end

	cb_insulin_stimulation_end = DiscreteCallback(condition_insulin_stimulation_end, affect_insulin_stimulation_end!, save_positions=(false, false))


	function condition_insulin_restimulation_start(u, t, integrator)
		t == 2880
	end

	function affect_insulin_restimulation_start!(integrator)
		integrator.u[103] = 499999.0
	end

	cb_insulin_restimulation_start = DiscreteCallback(condition_insulin_restimulation_start, affect_insulin_restimulation_start!, save_positions=(false, false))

	return CallbackSet(cb_insulin_restimulation_end, cb_insulin_stimulation_end, cb_insulin_restimulation_start), Function[], false
end


function computeTstops(u::AbstractVector, p::AbstractVector)
	return Float64[dualToFloat(2895.0), dualToFloat(p[68]), dualToFloat(2880.0)]
end
