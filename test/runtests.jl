using SafeTestsets


@safetestset "Aqua Quality Check" begin
  include("Aqua.jl")
end

@safetestset "PEtab-select" begin
  include("PEtab_select.jl")
end

@safetestset "Test model 2" begin
  include("Test_model2.jl")
end

@safetestset "Test model 3" begin
  include("Test_model3.jl")
end

@safetestset "Boehm against PyPesto" begin
  include("Boehm.jl")
end

@safetestset "PEtab remake" begin
  include("PEtab_remake.jl")
end

@safetestset "Log-likelihood values and gradients for published models" begin
  include("Test_ll.jl")
end

@safetestset "Utility functions" begin
  include(joinpath(@__DIR__, "Test_util.jl"))
end

@safetestset "Bijectors" begin
  include(joinpath(@__DIR__, "Bijectors.jl"))
end

@safetestset "PEtab test suite" begin
  include("PEtab_test_suite.jl")
end

@safetestset "Catalyst integration" begin
  include(joinpath(@__DIR__, "Julia_import", "Test_catalyst.jl"))
end

@safetestset "Model callibration" begin
  include(joinpath(@__DIR__, "Callibrate_model.jl"))
end

@safetestset "Optimisation results plotting" begin
  include(joinpath(@__DIR__, "Plot_optimisation_results.jl"))
end
