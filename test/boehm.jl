#=
    For the published Boehm model compare nllh, grad, and Gauss-Newton Hessian against
    pyPESTO computed values
=#

using PEtab, OrdinaryDiffEqRosenbrock, Sundials, SciMLSensitivity, CSV, DataFrames,
    LinearAlgebra, Test

include(joinpath(@__DIR__, "common.jl"))

function boehm_pyPESTO(model::PEtabModel, osolver::ODESolver)
    dirref = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014")
    xvals = CSV.read(joinpath(dirref, "Parameters_PyPesto.csv"), DataFrame;
                     drop=[:Id, :ratio, :specC17])
    ref_nllhs = CSV.read(joinpath(dirref, "Cost_PyPesto.csv"), DataFrame)[!, :Cost]
    ref_grads = CSV.read(joinpath(dirref, "Grad_PyPesto.csv"), DataFrame; drop=[:Id, :ratio, :specC17])
    ref_hessians = CSV.read(joinpath(dirref, "Hess_PyPesto.csv"), DataFrame)
    ih = findall( x -> x != "Id" && !occursin("ratio", x) && !occursin("specC17", x), names(ref_hessians))

    for i in 1:5
        x = xvals[i, :] |> Vector{Float64}
        nllh_ref = ref_nllhs[i]
        grad_ref = ref_grads[i, :] |> Vector{Float64}
        hess_ref = ref_hessians[i, :][ih] |> Vector{Float64}

        nllh = _compute_nllh(x, model, osolver)
        @test nllh ≈ nllh_ref atol=1e-3

        g = _compute_grad(x, model, :ForwardDiff, osolver)
        @test all(.≈(g, grad_ref; atol = 1e-3))
        g = _compute_grad(x, model, :ForwardEquations, osolver)
        @test all(.≈(g, grad_ref; atol = 1e-3))
        g = _compute_grad(x, model, :Adjoint, osolver;
                          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        @test all(.≈(g, grad_ref; atol = 1e-3))
        g = _compute_grad(x, model, :Adjoint, osolver;
                          sensealg=GaussAdjoint(autojacvec=ReverseDiffVJP(true)))
        @test all(.≈(g, grad_ref; atol = 1e-3))
        # Here we want to test things also run with CVODE_BDF
        tmp = osolver.solver
        osolver.solver = CVODE_BDF()
        g = _compute_grad(x, model, :ForwardEquations, osolver; sensealg = ForwardSensitivity())
        @test all(.≈(g, grad_ref; atol = 1e-3))
        osolver.solver = tmp

        H = _compute_hess(x, model, :GaussNewton, osolver)
        @test all(.≈(H[:], hess_ref; atol = 1e-3))
    end
end

path_yaml = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml)
@testset "Compare against pyPESTO" begin
    boehm_pyPESTO(model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end

# Test PEtab export
dir_export = joinpath(@__DIR__, "published_models", "Boehm_JProteomeRes2014", "tmp")
prob_export = PEtabModel(path_yaml) |>
    PEtabODEProblem
x_export = get_x(prob_export) .* 0.9
export_petab(dir_export, prob_export, x_export)
prob_exported = PEtabModel(joinpath(dir_export, basename(path_yaml))) |>
    PEtabODEProblem
@test get_x(prob_exported) == x_export
rm(dir_export; recursive = true)
