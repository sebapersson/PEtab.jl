using PEtab, Test

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
