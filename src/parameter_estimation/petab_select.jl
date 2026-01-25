"""
    petab_select(path_yaml, alg; nmultistarts = 100, kwargs...) -> path_res

For a PEtab-select problem, perform model selection with the method specified in the PEtab
select problem files. Returns the path (`path_res`) to a YAML-file with model selection
results.

The general model selection (e.g. to use forward-search) options are specified in the
PEtab-select files. For details on how to set this up, see the
PEtab-select [documentation](https://github.com/PEtab-dev/petab_select).

For each round of model selection, the candidate models are parameter estimated using
multi-start parameter estimation, with `nmultistarts` performed for each model.
The objective values obtained from parameter estimation are then used for the next round of
 model evaluation.

 A list of available and recommended optimization algorithms (`alg`) can be found in the
package documentation and [`calibrate`](@ref) documentation.

See also [`calibrate_multistart`](@ref).

## Keyword Arguments
- `kwargs`: The same keywords accepted by [`PEtabODEProblem`](@ref) and [`calibrate`](@ref).
"""
function petab_select end

# For developers, the actual code can be found in the PEtabSelectExt directory. This is
# just the way functions defined in extensions are handled.
