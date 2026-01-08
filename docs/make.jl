using Documenter
using DocumenterCitations
using PEtab

DocMeta.setdocmeta!(PEtab, :DocTestSetup, :(using PEtab); recursive = false)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"),
                           style=:numeric)

makedocs(;
    modules = [PEtab],
    repo = "https://github.com/sebapersson/PEtab.jl/blob/{commit}{path}#{line}",
    checkdocs = :exports,
    warnonly = false,
    plugins=[bib],
    sitename = "PEtab.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
        collapselevel = 1,
        repolink = "https://github.com/sebapersson/PEtab.jl",
        assets = String["assets/custom_theme.css"],),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Extended Tutorials" => Any[
            "Simulation conditions" => "define_problem/simulation_conditions.mdf",
            "Pre-equilibration (steady-state initialization)" => "define_problem/pre_equilibration.md",
            "Simulation condition-specific parameters" => "define_problem/condition_parameters.md",
            "Events/callbacks" => "define_problem/events.md",
            "Observable and noise parameters" => "define_problem/observable_noise_parameters.md",
            "Importing PEtab standard format" => "define_problem/standard_format.md"],
        "Parameter Estimation" => Any[
            "Parameter estimation tutorial" => "parameter_estimation/extended_tutorial.md",
            "Plotting parameter estimation results" => "parameter_estimation/plots.md",
            "Optimization algorithms and recommendations" => "parameter_estimation/optimizers.md",
            "Model selection with PEtab Select" => "parameter_estimation/model_selection.md",
            "Using optimizers directly" => "parameter_estimation/wrap.md"],
        "Bayesian inference" => "inference.md",
        "API" => "API.md",
        "Configuration and performance" => Any[
            "Default PEtabODEProblem options" => "configuration/default_options.md",
            "Derivative methods (gradients and Hessians)" => "configuration/derivatives.md",
            "Speeding up non-stiff models" => "performance/none_stiff.md",
            "Speeding up condition-specific parameters" => "performance/condition_parameters.md",
            "Speeding up large models with adjoints" => "Bachmann.md"
        ],
        "References" => "references.md",
        "FAQ" => "FAQ.md"],)

deploydocs(;
           repo = "github.com/sebapersson/PEtab.jl.git",
           devbranch = "main",)
