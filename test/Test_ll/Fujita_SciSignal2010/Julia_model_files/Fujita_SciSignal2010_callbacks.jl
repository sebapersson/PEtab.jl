function getCallbacks_Fujita_SciSignal2010()
	cb_EGF_bool1 = DiscreteCallback(condition_EGF_bool1, affect_EGF_bool1!, save_positions=(false, false))

	return CallbackSet(cb_EGF_bool1), [isActiveAtTime0_EGF_bool1!], false
end


function condition_EGF_bool1(u, t, integrator)
	t == integrator.p[1]
end

function affect_EGF_bool1!(integrator)
	integrator.p[6] = 1.0
end

function isActiveAtTime0_EGF_bool1!(u, p)
	t = 0.0 # Used to check conditions activated at t0=0
	p[6] = 0.0 # Default to being off
	if !(t ≤ p[1])
		p[6] = 1.0
	end
end


function computeTstops(u::AbstractVector, p::AbstractVector)
	 return Float64[dualToFloat(p[1])]
end