const MEASUREMENT_COLS = Dict("observableId" => (required = true, types = AbstractString),
                              "simulationConditionId" => (required = true,
                                                          types = AbstractString),
                              "measurement" => (required = true, types = Real),
                              "time" => (required = true, types = Real),
                              "preequilibrationConditionId" => (required = false,
                                                                types = Union{AbstractString,
                                                                              Missing}),
                              "observableParameters" => (required = false,
                                                         types = Union{AbstractString, Real,
                                                                       Missing}),
                              "noiseParameters" => (required = false,
                                                    types = Union{AbstractString, Real,
                                                                  Missing}))

const CONDITIONS_COLS = Dict("conditionId" => (required = true, types = AbstractString))

const PARAMETERS_COLS = Dict("parameterId" => (required = true, types = AbstractString),
                             "parameterScale" => (required = true, types = AbstractString),
                             "lowerBound" => (required = true, types = Union{Real, Missing}),
                             "upperBound" => (required = true, types = Union{Real, Missing}),
                             "nominalValue" => (required = true, types = Union{Real, AbstractString}),
                             "estimate" => (required = true, types = Real),
                             "initializationPriorType" => (required = false,
                                                           types = Union{Missing,
                                                                         AbstractString}),
                             "initializationPriorParameters" => (required = false,
                                                                 types = Union{Missing,
                                                                               AbstractString,
                                                                               Real}),
                             "objectivePriorType" => (required = false,
                                                      types = Union{Missing,
                                                                    AbstractString}),
                             "objectivePriorParameters" => (required = false,
                                                            types = Union{Missing,
                                                                          AbstractString,
                                                                          Real}))

const OBSERVABLES_COLS = Dict("observableId" => (required = true, types = AbstractString),
                              "observableFormula" => (required = true,
                                                      types = AbstractString),
                              "noiseFormula" => (required = true,
                                                 types = Union{AbstractString, Real}),
                              "observableTransformation" => (required = false,
                                                             types = Union{AbstractString,
                                                                           Missing}),
                              "noiseDistribution" => (required = false,
                                                      types = Union{AbstractString,
                                                                    Missing}))

const MAPPING_COLS = Dict("modelEntityId" => (required = true, types = AbstractString),
                          "petabEntityId" => (required = true, types = AbstractString))

const HYBRIDIZATION_COLS = Dict("targetId" => (required = true, types = AbstractString),
                                "targetValue" => (required = true, types = AbstractString))

const VALID_SCALES = ["lin", "log10", "log", "log2"]

# SBMLImporter is used for parsing functions in the PEtab syntax to Julia syntax
const PETAB_FUNCTIONS = Dict("pow" => SBMLImporter.FunctionSBML(["__x__", "__y__"],
                                                                "(__x__)^(__y__)"))
const PETAB_FUNCTIONS_NAMES = ["pow"]
