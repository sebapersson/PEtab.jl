using Test
using Distributions
using PEtab 
using CSV
using DataFrames
using Catalyst


@testset "Catalyst integration" begin
    include(joinpath(@__DIR__, "Case001.jl"))
    include(joinpath(@__DIR__, "Case002.jl"))
    include(joinpath(@__DIR__, "Case003.jl"))
    include(joinpath(@__DIR__, "Case004.jl"))
    include(joinpath(@__DIR__, "Case005.jl"))
    include(joinpath(@__DIR__, "Case006.jl"))
    include(joinpath(@__DIR__, "Case007.jl"))
    include(joinpath(@__DIR__, "Case008.jl"))
    include(joinpath(@__DIR__, "Case009.jl"))
    include(joinpath(@__DIR__, "Case010.jl"))
    include(joinpath(@__DIR__, "Case011.jl"))
    include(joinpath(@__DIR__, "Case013.jl"))
    include(joinpath(@__DIR__, "Case014.jl"))
    include(joinpath(@__DIR__, "Case015.jl"))
    include(joinpath(@__DIR__, "Case016.jl"))
    include(joinpath(@__DIR__, "Parameter_map.jl"))
    include(joinpath(@__DIR__, "Priors.jl"))
end

@testset "Testing input format" begin
    include(joinpath(@__DIR__, "Input_format.jl"))
end