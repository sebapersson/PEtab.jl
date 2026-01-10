using Documenter
using DocumenterCitations
using DocumenterVitepress
using PEtab

DocMeta.setdocmeta!(PEtab, :DocTestSetup, :(using PEtab); recursive = false)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"),
                           style=:numeric)

makedocs(
    modules = [PEtab],
    sitename = "PEtab.jl",
    repo = Remotes.GitHub("sebapersson", "PEtab.jl"),
    authors = "Sebastian Persson, and contributors",
    checkdocs = :exports,
    warnonly = false,
    plugins=[bib],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/sebapersson/PEtab.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "tutorial.md",
        "Tutorials" => Any[
            "Overview" => "tutorials/index.md",
            "Creating a parameter estimation problem" => Any[
                "Simulation conditions" => "tutorials/define_problem/simulation_conditions.md",
                "Pre-equilibration (steady-state initialization)" => "tutorials/define_problem/pre_equilibration.md",
                "Simulation condition-specific parameters" => "tutorials/define_problem/condition_parameters.md",
                "Events/callbacks" => "tutorials/define_problem/events.md",
                "Observable and noise parameters" => "tutorials/define_problem/observable_noise_parameters.md",
                "Importing PEtab standard format" => "tutorials/define_problem/standard_format.md",
            ],
            "Parameter estimation" => Any[
                "Parameter estimation tutorial" => "tutorials/parameter_estimation/extended_tutorial.md",
                "Plotting parameter estimation results" => "tutorials/parameter_estimation/plotting.md",
                "Optimization algorithms and recommendations" => "tutorials/parameter_estimation/optimizers.md",
                "Model selection with PEtab Select" => "tutorials/parameter_estimation/model_selection.md",
                "Using optimizers directly" => "tutorials/parameter_estimation/wrap.md",
                "Bayesian inference" => "tutorials/parameter_estimation/inference.md",  # if you want it here
            ],
        ],
        "API" => "API.md",
        "Configuration" => Any[
            "Default options" => "configuration/default_options.md",
            "Derivative methods (gradients and Hessians)" => "configuration/derivatives.md",
            "Speeding up non-stiff models" => "performance/none_stiff.md",
            "Speeding up condition-specific parameters" => "performance/condition_parameters.md",
            "Speeding up large models with adjoints" => "performance/adjoint.md",
        ],
        "References" => "references.md",
        "FAQ" => "FAQ.md",
    ],
)

DocumenterVitepress.deploydocs(
    repo = "github.com/sebapersson/PEtab.jl.git",
    target = "build", # this is where Vitepress stores its output
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)
