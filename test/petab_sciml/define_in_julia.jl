using Distributions
import PEtab: MLModel, PEtabMLParameter

include(joinpath(@__DIR__, "helper.jl"))

ode_solver = ODESolver(Rodas5P(), abstol = 1e-10, reltol = 1e-10, maxiters=Int(1e6))

@testset "SciML model in Julia" begin
    for i in 1:34
        test_case = i < 10 ? "00$(i).jl" : "0$(i).jl"
        include(joinpath(@__DIR__, test_case))
    end
end
