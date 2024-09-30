function get_reaction_system(foo)
	ModelingToolkit.@variables t
	D = Differential(t)
	sps = Catalyst.@species B(t) A(t) 
Any[]
	sps_arg = sps
	ps = Catalyst.@parameters b0 k1 a0 k2 __init__B__ compartment 

	_reactions = [
		Catalyst.Reaction(*(*(compartment, k1), A), [A], [B], [*(1.0, /(1.0, compartment))], [*(1.0, /(1.0, compartment))]; metadata = [:name => "fwd", :id => "fwd"], only_use_rate=true),
		Catalyst.Reaction(*(*(compartment, k2), B), [B], [A], [*(1.0, /(1.0, compartment))], [*(1.0, /(1.0, compartment))]; metadata = [:name => "rev", :id => "rev"], only_use_rate=true),
	]

	comb_ratelaws = false
	rn = Catalyst.ReactionSystem(_reactions, t, sps_arg, ps; name=Symbol("Conversion_Reaction_0"), combinatoric_ratelaws=comb_ratelaws)

	specie_map = [
	B =>__init__B__,
	A =>a0,
	]

	parameter_map = [
	b0 =>1.0,
	k1 =>0.0,
	a0 =>1.0,
	k2 =>0.0,
	__init__B__ =>b0,
	compartment =>1.0,
	]
	return rn, specie_map, parameter_map
end