include(joinpath(@__DIR__, "helper.jl"))
import PEtab: MLModel, PEtabMLParameter

ode_solver = ODESolver(Rodas5P(), abstol = 1e-10, reltol = 1e-10, maxiters=Int(1e6))

@testset "SciML model in Julia" begin
    for i in 1:31
        test_case = i < 10 ? "00$(i).jl" : "0$(i).jl"
        include(joinpath(@__DIR__, test_case))
    end
end
