#=
    Checking errors are thrown correctly for PEtab-SciML standard format
=#

import HDF5

dir_sciml_tests = joinpath(
    @__DIR__, "test_cases", "sciml_problem_import",
)
test_case = "014"

# Check metadata
# Change YAML to use tmp file
path_yaml = joinpath(dir_sciml_tests, test_case, "petab", "problem.yaml")
yaml_file = YAML.load_file(path_yaml)
yaml_file["extensions"]["sciml"]["array_files"] = ["net3_ps.hdf5", "net3_input1_tmp.hdf5"]
YAML.write_file(path_yaml, yaml_file)

# Create tmp file
path_hdf5 = joinpath(dir_sciml_tests, test_case, "petab", "net3_input1.hdf5")
path_hdf5_tmp = joinpath(dir_sciml_tests, test_case, "petab", "net3_input1_tmp.hdf5")
path_hdf5_tmp = cp(path_hdf5, path_hdf5_tmp; force = true)

# Test wrong PyTorch boolean
h5open(path_hdf5_tmp, "r+") do hdf5_file
    HDF5.write(hdf5_file["metadata"]["pytorch_format"], false)
end
@test_throws PEtab.PEtabInputError begin
    ml_models = MLModels(path_yaml)
    petab_model = PEtabModel(path_yaml; ml_models = ml_models)
end
# Test with no meta-data
h5open(path_hdf5_tmp, "r+") do hdf5_file
    HDF5.delete_object(hdf5_file["metadata"], "pytorch_format")
end
@test_throws PEtab.PEtabInputError begin
    ml_models = MLModels(path_yaml)
    petab_model = PEtabModel(path_yaml; ml_models = ml_models)
end

# Test with wrong input id
path_hdf5_tmp = cp(path_hdf5, path_hdf5_tmp; force = true)
h5open(path_hdf5_tmp, "r+") do hdf5_file
    HDF5.API.h5l_move(
        hdf5_file, "inputs/input0",
        hdf5_file, "inputs/my_new_input",
        HDF5.API.H5P_DEFAULT,
        HDF5.API.H5P_DEFAULT,
    )
end
@test_throws PEtab.PEtabInputError begin
    ml_models = MLModels(path_yaml)
    petab_model = PEtabModel(path_yaml; ml_models = ml_models)
end

# Done testing array file
yaml_file = YAML.load_file(path_yaml)
yaml_file["extensions"]["sciml"]["array_files"] = ["net3_ps.hdf5", "net3_input1.hdf5"]
YAML.write_file(path_yaml, yaml_file)
rm(path_hdf5_tmp)

# Test incorrect hybridization, gamma estimated
path_parameters = joinpath(dir_sciml_tests, test_case, "petab", "parameters.tsv")
df_parameters = CSV.read(path_parameters, DataFrame)
df_row = deepcopy(df_parameters[1, :])
df_row.parameterId = "gamma"
df_parameters = vcat(df_parameters, DataFrame(df_row))
CSV.write(path_parameters, df_parameters; delim = '\t')
@test_throws PEtab.PEtabInputError begin
    ml_models = MLModels(path_yaml)
    petab_model = PEtabModel(path_yaml; ml_models = ml_models)
end
df_parameters = CSV.read(path_parameters, DataFrame)
df_parameters = df_parameters[1:4, :]
CSV.write(path_parameters, df_parameters; delim = '\t')

# Test incorrect hybridization, not mapping to any model parameter
path_hybridization = joinpath(dir_sciml_tests, test_case, "petab", "hybridization.tsv")
df_hybridization = CSV.read(path_hybridization, DataFrame)
df_hybridization[2, :targetId] = "gamma1"
CSV.write(path_hybridization, df_hybridization; delim = '\t')
@test_throws PEtab.PEtabInputError begin
    ml_models = MLModels(path_yaml)
    petab_model = PEtabModel(path_yaml; ml_models = ml_models)
end
df_hybridization[2, :targetId] = "gamma"
CSV.write(path_hybridization, df_hybridization; delim = '\t')
