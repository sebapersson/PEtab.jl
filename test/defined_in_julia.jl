using Test, Distributions, PEtab, CSV, DataFrames, ModelingToolkit, Catalyst, LinearAlgebra,
      ForwardDiff

#  5 and 12 rely on SBML features and have no direct correspondence to
# ODESystem or Catalyst
@testset "Defined in Julia" begin
    include(joinpath(@__DIR__, "defined_in_julia", "001.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "002.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "003.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "004.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "006.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "007.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "008.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "009.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "010.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "011.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "013.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "014.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "015.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "016.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "017.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "018.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "020.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "log2.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "parametermap.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "priors.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "petab_events.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "input_check.jl"))
end
