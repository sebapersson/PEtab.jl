function get_reaction_system(foo)
	ModelingToolkit.@variables t
	D = Differential(t)
Any[]
	vs = ModelingToolkit.@variables A(t) B(t) 
	sps_arg = vs
	ps = Catalyst.@parameters __init__A__ b0 k1 a0 k2 __init__B__ compartment c1 

	_reactions = [
		D(A) ~ -(*(k2, B), *(k1, A)),
		D(B) ~ +(*(*(-(compartment), k2), B), *(*(compartment, k1), A)),
	]

	comb_ratelaws = true
	rn = Catalyst.ReactionSystem(_reactions, t, sps_arg, ps; name=Symbol("SBML_model"), combinatoric_ratelaws=comb_ratelaws)

	specie_map = [
	A =>__init__A__,
	B =>__init__B__,
	]

	parameter_map = [
	__init__A__ =>a0,
	b0 =>1.0,
	k1 =>0.0,
	a0 =>1.0,
	k2 =>0.0,
	__init__B__ =>b0,
	compartment =>1.0,
	c1 =>1.0,
	]
	return rn, specie_map, parameter_map
end