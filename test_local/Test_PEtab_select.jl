using PEtab
using OrdinaryDiffEq
using Optim 
using Test


# Why local test - need to connect to the correct python environment 
pathPythonExe = joinpath("/", "home", "sebpe", "anaconda3", "envs", "PeTab", "bin", "python")
ENV["PYTHON"] = pathPythonExe
import Pkg; Pkg.build("PyCall")


@testset "PEtab-select" begin

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0001", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0001", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, IPNewton())
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0002", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0002", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardDiff, hessianMethod=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0003", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0003", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0004", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0004", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0005", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0005", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0006", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0006", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0007", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0007", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3

    pathYAML = joinpath(@__DIR__, "PEtab_select", "0008", "petab_select_problem.yaml")
    pathExepected = joinpath(@__DIR__, "PEtab_select", "0008", "expected.yaml")
    pathSave = runPEtabSelect(pathYAML, Fides(verbose=false), gradientMethod=:ForwardEquations, hessianMethod=:GaussNewton, reuseS=true, sensealg=:ForwardDiff, nOptimisationStarts=10)
    dataExpected = YAML.load_file(pathExepected)
    dataComputed = YAML.load_file(pathSave)
    @test dataExpected["criteria"]["NLLH"] ≈ dataComputed["criteria"]["NLLH"] atol=1e-3
end
