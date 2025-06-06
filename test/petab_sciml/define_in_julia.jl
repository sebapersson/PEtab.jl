include(joinpath(@__DIR__, "helper.jl"))

@testset "SciML model in Julia" begin
    for i in 1:28
        i in [14, 15] && continue
        test_case = i < 10 ? "00$(i).jl" : "0$(i).jl"
        include(joinpath(@__DIR__, test_case))
    end
end
