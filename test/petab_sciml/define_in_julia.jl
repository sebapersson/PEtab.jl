include(joinpath(@__DIR__, "helper.jl"))

@testset "SciML model in Julia" begin
    include(joinpath(@__DIR__, "001.jl"))
    include(joinpath(@__DIR__, "002.jl"))
    include(joinpath(@__DIR__, "003.jl"))
    include(joinpath(@__DIR__, "004.jl"))
    include(joinpath(@__DIR__, "005.jl"))
    include(joinpath(@__DIR__, "006.jl"))
end
