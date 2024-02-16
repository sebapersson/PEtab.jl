function get_reaction_system(foo)

	ModelingToolkit.@variables t
	sps = Catalyst.@species IR2(t) IR2in(t) Rec2(t) IR1in(t) Uptake1(t) Uptake2(t) InsulinFragments(t) IR1(t) Rec1(t) Ins(t) BoundUnspec(t) 
	ps = Catalyst.@parameters ka1 ini_R2fold kout ini_R1 kout_frag koff_unspec kin ka2fold kin2 kd1 kon_unspec init_Ins kd2fold kout2 ExtracellularMedium 

	D = Differential(t)

	_reactions = [
		Catalyst.Reaction((ExtracellularMedium*IR1in)*kout, [IR1in], [IR1], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction(((ExtracellularMedium*Ins)*Rec1)*ka1, [Ins, Rec1], [IR1], [1.0/ExtracellularMedium, 1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*IR2in)*kout2, [IR2in], [IR2], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*IR2)*kin2, [IR2], [IR2in], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction(ExtracellularMedium*((((Ins*Rec2)*ka1)*ka2fold)-((IR2*kd1)*kd2fold)), nothing, [Uptake2], nothing, [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*IR1)*kin, [IR1], [IR1in], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((((ExtracellularMedium*Ins)*Rec2)*ka1)*ka2fold, [Ins, Rec2], [IR2], [1.0/ExtracellularMedium, 1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*IR1)*kd1, [IR1], [Ins, Rec1], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium, 1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction(((ExtracellularMedium*IR2)*kd1)*kd2fold, [IR2], [Ins, Rec2], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium, 1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*Ins)*kon_unspec, [Ins], [BoundUnspec], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*IR1in)*kout_frag, [IR1in], [Rec1, InsulinFragments], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium, 1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*BoundUnspec)*koff_unspec, [BoundUnspec], [Ins], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction((ExtracellularMedium*IR2in)*kout_frag, [IR2in], [Rec2, InsulinFragments], [1.0/ExtracellularMedium], [1.0/ExtracellularMedium, 1.0/ExtracellularMedium]; only_use_rate=true),
		Catalyst.Reaction(ExtracellularMedium*(((Ins*Rec1)*ka1)-(IR1*kd1)), nothing, [Uptake1], nothing, [1.0/ExtracellularMedium]; only_use_rate=true),
	]


	rn = Catalyst.ReactionSystem(_reactions, t, sps, ps; name=Symbol("Schwen_PONE2014"), combinatoric_ratelaws=false)

	specie_map = [
	IR2 =>0.0,
	IR2in =>0.0,
	Rec2 =>ini_R1*ini_R2fold,
	IR1in =>0.0,
	Uptake1 =>0.0,
	Uptake2 =>0.0,
	InsulinFragments =>0.0,
	IR1 =>0.0,
	Rec1 =>ini_R1,
	Ins =>init_Ins,
	BoundUnspec =>0.0,
	]
	parameter_map = [
	ka1 =>0.00937980436663883,
	ini_R2fold =>16.457631927604,
	kout =>324.725838145278,
	ini_R1 =>47.2370172820096,
	kout_frag =>0.0100421669378689,
	koff_unspec =>9.6457882009227,
	kin =>3.75654890101743,
	ka2fold =>2.0907692381484,
	kin2 =>0.545304029714509,
	kd1 =>6.72269169161034,
	kon_unspec =>19.941427249128,
	init_Ins =>0.0,
	kd2fold =>9.61850107655493,
	kout2 =>0.0529079560976487,
	ExtracellularMedium =>1.0,
	]
	return rn, specie_map, parameter_map
end