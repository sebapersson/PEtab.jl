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
        "Bayesian Inference" => "inference.md",
        "API" => "API.md",
        "Gradient and Hessian Methods" => "grad_hess_methods.md",
        "Default Options" => "default_options.md",
        "Performance Tips" => Any[
            "Non-Biology (Non-Stiff) Models" => "nonstiff_models.md",
            "Condition-Specific Parameters" => "Beer.md",
            "Adjoint Sensitivity Analysis (Large Models)" => "Bachmann.md"],
        "References" => "references.md",
        "FAQ" => "FAQ.md"],)

deploydocs(;
           repo = "github.com/sebapersson/PEtab.jl.git",
           devbranch = "main",)
