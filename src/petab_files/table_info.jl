const MEASUREMENT_V1_COLS = Dict(
    "observableId" => (required = true, types = AbstractString),
    "simulationConditionId" => (required = true, types = AbstractString),
    "measurement" => (required = true, types = Real),
    "time" => (required = true, types = Real),
    "preequilibrationConditionId" => (required = false,
                                      types = Union{AbstractString, Missing}),
    "observableParameters" => (required = false,
                               types = Union{AbstractString, Real, Missing}),
    "noiseParameters" => (required = false,
                          types = Union{AbstractString, Real, Missing})
)

const CONDITIONS_V1_COLS = Dict(
    "conditionId" => (required = true, types = AbstractString)
)

const PARAMETERS_V1_COLS = Dict(
    "parameterId" => (required = true, types = AbstractString),
    "parameterScale" => (required = true, types = AbstractString),
    "lowerBound" => (required = true, types = Union{Real, Missing}),
    "upperBound" => (required = true, types = Union{Real, Missing}),
    "nominalValue" => (required = true, types = Real),
    "estimate" => (required = true, types = Real),
    "initializationPriorType" => (required = false, types = Union{Missing, AbstractString}),
    "initializationPriorParameters" => (required = false,
                                        types = Union{Missing, AbstractString}),
    "objectivePriorType" => (required = false,
                             types = Union{Missing, AbstractString}),
    "objectivePriorParameters" => (required = false,
                                  types = Union{Missing, AbstractString})
)

const OBSERVABLES_V1_COLS = Dict(
    "observableId" => (required = true, types = AbstractString),
    "observableFormula" => (required = true, types = AbstractString),
    "noiseFormula" => (required = true, types = Union{AbstractString, Real}),
    "observableTransformation" => (required = false,
                                  types = Union{AbstractString, Missing}),
    "noiseDistribution" => (required = false, types = Union{AbstractString, Missing})
)

const MEASUREMENT_V2_COLS = Dict(
    "observableId" => (required = true, types = AbstractString),
    "experimentId" => (required = true, types = Union{AbstractString, Missing}),
    "measurement" => (required = true, types = Real),
    "time" => (required = true, types = Real),
    "observableParameters" => (required = false,
                               types = Union{AbstractString, Real, Missing}),
    "noiseParameters" => (required = false,
                          types = Union{AbstractString, Real, Missing})
)

const CONDITIONS_V2_COLS = Dict(
    "conditionId" => (required = true, types = AbstractString),
    "targetId" => (required = true, types = AbstractString),
    "targetValue" => (required = true, types = Union{AbstractString, Real})
)

const OBSERVABLES_V2_COLS = Dict(
    "observableId" => (required = true, types = AbstractString),
    "observableFormula" => (required = true, types = AbstractString),
    "noiseFormula" => (required = true, types = Union{AbstractString, Real}),
    "noiseDistribution" => (required = false, types = Union{AbstractString, Missing}),
    "noisePlaceholders" => (required = false, types = Union{AbstractString, Missing}),
    "observablePlaceholders" => (required = false, types = Union{AbstractString, Missing})
)

const PARAMETERS_V2_COLS = Dict(
    "parameterId" => (required = true, types = AbstractString),
    "parameterScale" => (required = false, types = AbstractString),
    "lowerBound" => (required = true, types = Union{Real, Missing}),
    "upperBound" => (required = true, types = Union{Real, Missing}),
    "nominalValue" => (required = true, types = Real),
    "estimate" => (required = true, types = Real),
    "priorDistribution" => (required = false, types = Union{Missing, AbstractString}),
    "priorParameters" => (required = false,
                          types = Union{Missing, Real, AbstractString})
)

const EXPERIMENTS_V2_COLS = Dict(
    "experimentId" => (required = true, types = AbstractString),
    "time" => (required = true, types = Real),
    "conditionId" => (required = true, types = Union{AbstractString, Missing})
)

const COLUMN_INFO = Dict(
    :measurements_v1 => MEASUREMENT_V1_COLS,
    :conditions_v1 => CONDITIONS_V1_COLS,
    :parameters_v1 => PARAMETERS_V1_COLS,
    :observables_v1 => OBSERVABLES_V1_COLS,
    :experiments_v2 => EXPERIMENTS_V2_COLS,
    :measurements_v2 => MEASUREMENT_V2_COLS,
    :conditions_v2 => CONDITIONS_V2_COLS,
    :parameters_v2 => PARAMETERS_V2_COLS,
    :observables_v2 => OBSERVABLES_V2_COLS
)

const VALID_SCALES = ["lin", "log10", "log", "log2"]

# SBMLImporter is used for parsing functions in the PEtab syntax to Julia syntax
const PETAB_FUNCTIONS = Dict("pow" => SBMLImporter.FunctionSBML(["__x__", "__y__"],
                                                                "(__x__)^(__y__)"))
const PETAB_FUNCTIONS_NAMES = ["pow"]
