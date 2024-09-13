function get_reaction_system(foo)
	ModelingToolkit.@variables t
	D = Differential(t)
	sps = Catalyst.@species x(t) y(t) 
Any[]
	sps_arg = sps
	ps = Catalyst.@parameters c b a_scale a d default 

	_reactions = [
		Catalyst.Reaction(+(-(*(a, a_scale), *(b, x)), *(c, y)), nothing, [x], nothing, [*(1.0, /(1.0, default))]; metadata = [:name => "v_0", :id => "v1_v_0"], only_use_rate=true),
		Catalyst.Reaction(-(*(b, x), +(*(c, y), *(d, y))), nothing, [y], nothing, [*(1.0, /(1.0, default))]; metadata = [:name => "v_1", :id => "v2_v_1"], only_use_rate=true),
	]

	comb_ratelaws = false
	rn = Catalyst.ReactionSystem(_reactions, t, sps_arg, ps; name=Symbol("Test_model3"), combinatoric_ratelaws=comb_ratelaws)

	specie_map = [
	x =>0.0,
	y =>0.0,
	]

	parameter_map = [
	c =>3.0,
	b =>2.0,
	a_scale =>1.0,
	a =>1.0,
	d =>4.0,
	default =>1.0,
	]
	return rn, specie_map, parameter_map
end