#=
    With SciMLStructures the indexing should get easier, but we cannot commit to this
    yet as SciMLSensitivity does not integrate with SciMLStructures. Then, how do we
    handle this?
    Input user the user can retreive later should work with the MTK interface, to keep
    in mind for the util funcitons.
    Once SciMLStructures are the prime soluation in SciMLSensitivity the internal mappings
    will change with the help of functions like setu, setp etc..., but for the moment I
    will use the internal PEtab ones.
    SciMLStructures is still the main goal, especially I find the constant feature to be
    very desired. I think MTKParameters might be sufficient for our needs, but
    will have to wait. This will complicate the callback handling a bit, but not really
    that much (will add funciton whether or not SciMLStructures are supported as a flag)
=#
using SciMLStructures, SBMLImporter, OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity
path_SBML = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "model_Boehm_JProteomeRes2014.xml")
prn, cbs = load_SBML(path_SBML)

tspan = (0.0, 10.0)
sys = structural_simplify(convert(ODESystem, prn.rn))
oprob = ODEProblem(sys, prn.u0, tspan, prn.p, jac=true)
mysetp = ModelingToolkit.setp(oprob, [:Epo_degradation_BaF3, :cyt])
oprob_new = remake(oprob, p = oprob.p.tunable |> deepcopy)

solve(oprob, Rodas5P(), abstol = 1e-14, reltol = 1e-14, saveat = [10.0])
solve(oprob_new, Rodas5P(), abstol = 1e-14, reltol = 1e-14, saveat = [10.0])

oprob_new.p[[3, 1]] .= 1.0
mysetp(oprob, [1.0, 1.0])
solve(oprob, Rodas5P(), abstol = 1e-14, reltol = 1e-14, saveat = [10.0])
solve(oprob_new, Rodas5P(), abstol = 1e-14, reltol = 1e-14, saveat = [10.0])

function mycost(x::Vector{T})::T where T <: Real
    newp = SciMLStructures.replace(SciMLStructures.Tunable(), oprob.p,
                                   convert.(eltype(x), oprob.p.tunable))
    oprob_new = remake(oprob, p = newp, u0 = convert.(eltype(x), oprob.u0))
    mysetp(oprob_new, x)
    sol = solve(oprob_new, Rodas5P(), abstol = 1e-14, reltol = 1e-14, saveat = [10.0])
    return sum(sol[:STAT5A] .* 0.01)
end


# To not break
oprob_adj = remake(oprob, p = oprob.p.tunable)
sol = solve(oprob_adj, Rodas5P(), abstol = 1e-14, reltol = 1e-14)
dg(out,u,p,t,i) = (out.=-1.0.+u)
ts = 0:5.0:10
res = adjoint_sensitivities(sol, Rodas5P(autodiff = false); t=ts, dg_discrete=dg, abstol=1e-14,
                            reltol=1e-14, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))

newprob = ODEForwardSensitivityProblem(oprob.f, oprob.u0, oprob.tspan, oprob.p.tunable,
                                       sensealg = ForwardDiffSensitivity())
sol = solve(newprob, Rodas4P(autodiff = false), abstol = 1e-8, reltol = 1e-8)

x = Float32[3.0, 4.0]

foo = SciMLStructures.replace(SciMLStructures.Tunable, oprob.p,  oprob.p.tunable .|> eltype(x))

oprob = remake(oprob; p = [:Epo_degradation_BaF3 => 5.0])
remake(oprob.p)
