include(joinpath(@__DIR__, "helper.jl"))

@testset "SciML model in Julia" begin
    include(joinpath(@__DIR__, "001.jl"))
    include(joinpath(@__DIR__, "002.jl"))
    include(joinpath(@__DIR__, "003.jl"))
    include(joinpath(@__DIR__, "004.jl"))
    include(joinpath(@__DIR__, "005.jl"))
    include(joinpath(@__DIR__, "006.jl"))
    include(joinpath(@__DIR__, "007.jl"))
    include(joinpath(@__DIR__, "008.jl"))
    include(joinpath(@__DIR__, "009.jl"))
    include(joinpath(@__DIR__, "010.jl"))
    include(joinpath(@__DIR__, "011.jl"))
    include(joinpath(@__DIR__, "012.jl"))
    include(joinpath(@__DIR__, "013.jl"))
    include(joinpath(@__DIR__, "014.jl"))
    include(joinpath(@__DIR__, "015.jl"))
    include(joinpath(@__DIR__, "016.jl"))
end
