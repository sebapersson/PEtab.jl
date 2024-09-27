"""
    get_obs_comparison_plots(res, prob::PEtabODEProblem; kwargs...)::Dict

Plot the model fit against data for all simulation conditions and all observable ids for
a PEtab.jl parameter estimation result (`res`).

Each entry in the returned `Dict` corresponds to
`plot(res, prob; obsids=[obsid], cid=cid, kwargs...)` for all possible `cid` and
all `obsid`.
"""
function get_obs_comparison_plots end
