function get_reaction_system(foo)

	ModelingToolkit.@variables t
	sps = Catalyst.@species x1(t) x2(t) 
	vs = ModelingToolkit.@variables observable_x2(t)
	ps = Catalyst.@parameters k3 k1 k2 default 

	D = Differential(t)

	_reactions = [
		Catalyst.Reaction(default*((((k3)*(x1))*(x2))/(default)), [x1, x2], nothing, [1.0/default, 1.0/default], nothing; only_use_rate=true),
		Catalyst.Reaction(default*((k1)/(default)), nothing, [x1], nothing, [1.0/default]; only_use_rate=true),
		Catalyst.Reaction(default*(((k2)*(x1))/(default)), [x1], [x2], [1.0/default], [1.0/default]; only_use_rate=true),
		D(observable_x2) ~ 0.0,
	]


	rn = Catalyst.ReactionSystem(_reactions, t, [sps; vs], ps; name=Symbol("caroModel"), combinatoric_ratelaws=false)

	specie_map = [
	x1 =>0.0,
	x2 =>0.0,
	observable_x2 => 0.0,
	]
	parameter_map = [
	k3 =>0.0,
	k1 =>0.2,
	k2 =>0.1,
	default =>1.0,
	]
	return rn, specie_map, parameter_map
end