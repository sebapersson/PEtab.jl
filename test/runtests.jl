using SafeTestsets

core_only = get(ENV, "CORE_ONLY", "false") == "true"

@safetestset "Aqua Quality Check" begin
  include("aqua.jl")
end

@safetestset "Analytic solution test model" begin
    include("analytic_solution.jl")
end

@safetestset "PEtab v1 test suite" begin
    include("petab_v1_testsuite.jl")
end

@safetestset "PEtab v2 test suite" begin
    include("petab_v2_testsuite.jl")
end

@safetestset "Define models in Julia" begin
    include("defined_in_julia.jl")
end

@safetestset "Parameter estimation" begin
    include("calibrate_core.jl")
end

@safetestset "Optimization results plotting" begin
    include("plot_optimisation_results.jl")
end

@safetestset "Utility functions" begin
    include(joinpath(@__DIR__, "util.jl"))
end

@safetestset "Log-Laplace" begin
    include(joinpath(@__DIR__, "log_laplace.jl"))
end

if !core_only
    @safetestset "Model with analytic steady-state" begin
        include("analytic_ss.jl")
    end

    @safetestset "Boehm model with pyPESTO reference" begin
        include("boehm.jl")
    end

    @safetestset "PEtab remake" begin
    include("remake.jl")
    end

    @safetestset "Log-likelihood values and gradients for published models" begin
        include("published_models.jl")
    end

    @safetestset "Component Arrays" begin
        include("component_arrays.jl")
    end

    @safetestset "Default options" begin
        include("defaults.jl")
    end

    @safetestset "Show" begin
        include("show.jl")
    end

    @safetestset "Parameter estimation expanded" begin
        include("calibrate.jl")
    end

    @safetestset "Bijectors" begin
        include("bijectors.jl")
    end

    @safetestset "Bayesian Inference" begin
        include("inference.jl")
    end

    @safetestset "PEtab-select" begin
        include("petab_select_testsuite.jl")
    end
end
