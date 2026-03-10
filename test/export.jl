using Lux, PEtab, Test

# v1 problem
path_yaml = joinpath(
    @__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"
)
dir_export = joinpath(
    @__DIR__, "published_models", "Boehm_JProteomeRes2014", "tmp"
)
prob_export = PEtabModel(path_yaml) |>
    PEtabODEProblem
x_export = get_x(prob_export) .* 0.9
export_petab(dir_export, prob_export, x_export)
prob_exported = PEtabModel(joinpath(dir_export, basename(path_yaml))) |>
    PEtabODEProblem
@test get_x(prob_exported) == x_export
rm(dir_export; recursive = true)

# v2 problem
path_yaml = joinpath(@__DIR__, "petab_v2_testsuite", "0002", "_0002.yaml")
dir_export = joinpath(@__DIR__, "petab_v2_testsuite", "0002", "tmp")
prob_export = PEtabModel(path_yaml) |>
    PEtabODEProblem
x_export = get_x(prob_export) .* 0.9
export_petab(dir_export, prob_export, x_export)
prob_exported = PEtabModel(joinpath(dir_export, basename(path_yaml))) |>
    PEtabODEProblem
@test get_x(prob_exported) == x_export
rm(dir_export; recursive = true)

# PEtab-SciML problem
path_yaml = joinpath(
    @__DIR__, "petab_sciml_testsuite", "test_cases", "sciml_problem_import", "001",
    "petab", "problem.yaml"
)
dir_export = joinpath(@__DIR__, "petpetab_sciml_testsuiteb_v2_testsuite", "export_tmp")
ml_models = MLModels(path_yaml)
prob_export = PEtabModel(path_yaml; ml_models = ml_models) |>
    PEtabODEProblem
x_export = get_x(prob_export) .* 0.9
export_petab(dir_export, prob_export, x_export)
path_exported = joinpath(dir_export, basename(path_yaml))
prob_exported = PEtabModel(path_exported; ml_models = MLModels(path_exported)) |>
    PEtabODEProblem
@test get_x(prob_exported) == x_export
rm(dir_export; recursive = true)
