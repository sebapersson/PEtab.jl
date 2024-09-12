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
                             "lowerBound" => (required = true, types = Real),
                             "upperBound" => (required = true, types = Real),
                             "nominalValue" => (required = true, types = Real),
                             "estimate" => (required = true, types = Real),
                             "initializationPriorType" => (required = false,
                                                           types = Union{Missing,
                                                                         AbstractString}),
                             "initializationPriorParameters" => (required = false,
                                                                 types = Union{Missing,
                                                                               AbstractString}),
                             "objectivePriorType" => (required = false,
                                                      types = Union{Missing,
                                                                    AbstractString}),
                             "objectivePriorParameters" => (required = false,
                                                            types = Union{Missing,
                                                                          AbstractString}))

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

const VALID_SCALES = ["lin", "log10", "log"]

# SBMLImporter is used for parsing functions in the PEtab syntax to Julia syntax
const PETAB_FUNCTIONS = Dict("pow" => SBMLImporter.FunctionSBML(["__x__", "__y__"], "(__x__)^(__y__)"))
const PETAB_FUNCTIONS_NAMES = ["pow"]
