"""
    UDEProblem(f!, u0, tspan, p_mechanistic, ml_models)

Create an `ODEProblem` for universal differential equation (UDE) models combining mechanistic
and machine-learning components.

`UDEProblem` wraps `f!` such that `ml_models` is available during RHS evaluation and
constructs the full parameter vector by combining `p_mechanistic` with ML parameters for
each `MLModel` in `ml_models`.

Both `u0` and `p_mechanistic` must be a NamedTuple or ComponentArray such that model states
and parameters can be indexed by id for PEtab.jl to correctly map parameters and initial
values.

For examples, see the online package documentation.

# Arguments
- `f!`: In-place RHS function. Must support the signature `f!(du, u, p, t, ml_models)`.
- `u0::Union{NamedTuple, ComponentArray}`: Initial condition. Must be named such that model
  states can be indexed by id (e.g. `u[:prey]`).
- `tspan`: Simulation time span `(t0, tf)`.
- `p_mechanistic::Union{NamedTuple, ComponentArray}`: Mechanistic parameters. Must be named
  such that parameters can be indexed by id (e.g. `p[:alpha]`).
- `ml_models::Union{MLModel, MLModels}`: ML model(s) used by `f!`.
"""
function UDEProblem(
        f!::Function, u0::Union{NamedTuple, ComponentArray}, tspan,
        p_mechanistic::Union{NamedTuple, ComponentArray},
        ml_models::Union{MLModel, MLModels}
    )::ODEProblem
    # Check `f!` has a 5-argument method: f!(du, u, p, t, ml_models)
    if !any(m -> m.nargs == 6, methods(f!))
        throw(ArgumentError("`f!` must have a method with signature `f!(du, u, p, t, \
            ml_models)`."))
    end

    ml_models = ml_models isa MLModel ? MLModels(ml_models) : ml_models

    xnominal_ml_models = Vector{ComponentArray}(undef, length(ml_models.ml_ids))
    for (i, ml_model) in pairs(ml_models.ml_models)
        xnominal_ml_models[i] = _get_lux_ps(ComponentArray, ml_model)
    end
    x_ml_models = NamedTuple(ml_models.ml_ids .=> xnominal_ml_models)
    p_ode = merge(NamedTuple(p_mechanistic), x_ml_models) |>
        ComponentArray .|>
        Float64
    u0 = ComponentArray(u0) .|>
        Float64

    f_ode! = let _ml_models = ml_models
        (du, u, p, t) -> f!(du, u, p, t, _ml_models)
    end
    return ODEProblem(f_ode!, u0, tspan, p_ode)
end

"""
    _get_ode_ml_models(ml_models, path_SBML, petab_tables)

Identify which neural-network models appear in the ODE right-hand-side

For this to hold, the following must hold:

1. Neural network has static = false
2. All neural network inputs and outputs appear in the hybridization table
3. All neural network outputs should assign to SBML model parameters

In case 3 does not hold, an error should be thrown as something is wrong with the PEtab
problem.
"""
function _get_ode_ml_models(
        ml_models::MLModels, path_SBML::String, petab_tables::PEtabTables
    )::MLModels
    isempty(ml_models) && return MLModels()

    out = MLModel[]
    libsbml_model = SBMLImporter.SBML.readSBML(path_SBML)
    hybridization_df, mappings_df = _get_petab_tables(
        petab_tables, [:hybridization, :mapping]
    )

    # Sanity check that columns in mapping table are correctly named
    pattern = r"(.inputs|.outputs|.parameters)"
    for io_id in string.(mappings_df[!, "modelEntityId"])
        if !occursin(pattern, io_id)
            throw(PEtabInputError("In mapping table, in modelEntityId column allowed \
                                   values are only ml_id.inputs..., ml_id.outputs... \
                                   and ml_id.parameters... $io_id is invalid"))
        end
    end

    for ml_model in ml_models.ml_models
        ml_model.static == true && continue

        @unpack ml_id = ml_model
        input_variables = Iterators.flatten(
            _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
        )
        if !all([x in hybridization_df.targetId for x in input_variables])
            throw(PEtab.PEtabInputError("For a static=false neural network all input \
                must be assigned value in the hybridization table. This does not hold for \
                $ml_id"))
        end

        output_variables = Iterators.flatten(
            _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs)
        )
        outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
        isempty(outputs_df) && continue

        if !all([x in keys(libsbml_model.parameters) for x in outputs_df.targetId])
            throw(PEtab.PEtabInputError("For a static=false neural network all output \
                variables in hybridization table must map to SBML model parameters. This \
                does not hold for $ml_id"))
        end

        push!(out, ml_model)
    end
    return MLModels(out)
end

function _get_ml_ids(mappings_df::DataFrame)::Vector{String}
    isempty(mappings_df) && return String[]
    return split.(string.(mappings_df[!, "modelEntityId"]), ".") .|>
        first .|>
        string
end

function _get_ml_model_indices(
        ml_id::Symbol, mapping_table_ids::Vector{String}
    )::Vector{Int64}
    ix = findall(x -> startswith(x, string(ml_id)), mapping_table_ids)
    return sort(ix, by = i -> count(c -> c == '[' || c == '.', mapping_table_ids[i]))
end
