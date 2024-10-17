using Test, PEtab, StochasticDiffEq

path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml)
petab_prob = PEtabODEProblem(model)

x = get_x(petab_prob)
sol = get_odesol(x, petab_prob)
tsave = petab_prob.model_info.simulation_info.tsaves[:model1_data1]
minfo = PEtab.MeasurementsInfo(petab_prob.model_info, :model1_data1)

llh = 0.0
for (it, t) in pairs(minfo.t)
    PEtab._set_x_minfo!(minfo, x, petab_prob)
    u = sol(t)
    p = sol.prob.p
    llh += PEtab.llh(u, p, it, minfo)
end
nllh = petab_prob.nllh(x)
@test nllh ≈ llh * -1
