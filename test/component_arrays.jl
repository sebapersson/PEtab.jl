#=
    Ideally PEtab wants the input vector to be a ComponentArray, but it can also accept
    vector input. This file tests that both vector and ComponentArray input works.
=#

using PEtab, Test

path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml)
prob = PEtabODEProblem(model)
xnames_ps = prob.model_info.xindices.ids[:estimate_ps]
xnames_ps_rev = reverse(propertynames(prob.xnominal_transformed))

x_ca = prob.xnominal_transformed
x_ca_rev = x_ca[xnames_ps_rev]
x = x_ca |> collect
nllh_ref = prob.nllh(x)
nllh_ca = prob.nllh(x_ca)
@test nllh_ref == nllh_ca
@test_throws PEtab.PEtabInputError begin
    prob.nllh(x_ca_rev)
end

grad_ref = prob.grad(x)
grad_ca = prob.grad(x_ca)
@test grad_ref == collect(grad_ca)

hess_ref = prob.hess(x)
hess_ca = prob.hess(x_ca)
hess_ca_rev = prob.hess(x_ca_rev)
@test hess_ref == hess_ca
