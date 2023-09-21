function getCallbacks_Brannmark()

	function condition_insulin_bool1(u, t, integrator)
		t - integrator.p[10] == 0
	end

	function affect_insulin_bool1!(integrator)
		integrator.p[3] = 1.0
	end

	function is_active_t0_insulin_bool1!(u, p)
		t = 0.0 # Used to check conditions activated at t0=0
		p[3] = 0.0 # Default to being off
		if !(t - p[10] < 0)
			p[3] = 1.0
		end
	end


	cb_insulin_bool1 = DiscreteCallback(condition_insulin_bool1, affect_insulin_bool1!, save_positions=(false, false))


	function condition_insulin_bool2(u, t, integrator)
		t - integrator.p[11] == 0
	end

	function affect_insulin_bool2!(integrator)
		integrator.p[14] = 1.0
	end

	function is_active_t0_insulin_bool2!(u, p)
		t = 0.0 # Used to check conditions activated at t0=0
		p[14] = 0.0 # Default to being off
		if !(t - p[11] < 0)
			p[14] = 1.0
		end
	end


	cb_insulin_bool2 = DiscreteCallback(condition_insulin_bool2, affect_insulin_bool2!, save_positions=(false, false))

	return CallbackSet(cb_insulin_bool1, cb_insulin_bool2), Function[is_active_t0_insulin_bool1!, is_active_t0_insulin_bool2!], false
end


function computeTstops(u::AbstractVector, p::AbstractVector)
	return Float64[dual_to_float(p[10]), dual_to_float(p[11])]
end
