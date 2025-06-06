using Catalyst, CSV, DataFrames, Distributions, ForwardDiff, ModelingToolkit,
    OrdinaryDiffEqRosenbrock, PEtab, Test

#  5 and 12 rely on SBML features and have no direct correspondence to a MTK ODESystem or
# Catalyst ReactionSystem
@testset "Defined in Julia" begin
    for i in 1:18
        i in [5, 12] && continue
        @info "Test case $i"
        test_case = i < 10 ? "00$(i).jl" : "0$(i).jl"
        include(joinpath(@__DIR__, "defined_in_julia", test_case))
    end
    include(joinpath(@__DIR__, "defined_in_julia", "log2.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "parametermap.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "priors.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "petab_events.jl"))
    include(joinpath(@__DIR__, "defined_in_julia", "input_check.jl"))
end
