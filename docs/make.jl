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
                                  repolink = "https://github.com/sebapersson/PEtab.jl",
                                  assets = String["assets/custom_theme.css"],),
         pages = ["Home" => "index.md",
                  "Tutorial" => "tutorial.md",
                  "Extended Tutorials" => Any["Simulation conditions" => "petab_simcond.md",
                                              "Steady-State Simulations (Pre-Equilibration)" => "petab_preeq_simulations.md",
                                              "Events (callbacks, dosages, etc.)" => "petab_event.md",
                                              "Simulation Condition-Specific Parameters" => "petab_cond_specific.md",
                                              "Noise and Observable Parameters" => "petab_obs_noise.md",
                                              "Importing PEtab Standard Format" => "import_petab.md"],
                  "Parameter Estimation" => Any["Parameter Estimation Methods" => "pest_method.md",
                                                "Plotting Estimation Results" => "pest_plot.md",
                                                "Available and Recommended Algorithms" => "pest_algs.md",
                                                "Model Selection with PEtab-select" => "pest_select.md",
                                                "Wrapping Optimization Packages" => "pest_custom.md"],
                  "Bayesian Inference" => "inference.md",
                  "API" => "API.md",
                  "Gradient and Hessian Methods" => "grad_hess_methods.md",
                  "Default Options" => "default_options.md",
                  "Performance Tips" => Any["Non-Biology (Non-Stiff) Models" => "nonstiff_models.md",
                                            "Condition-Specific Parameters" => "Beer.md",
                                            "Adjoint Sensitivity Analysis (Large Models)" => "Bachmann.md"],
                  "References" => "references.md",
                  "FAQ" => "FAQ.md"],)

deploydocs(;
           repo = "github.com/sebapersson/PEtab.jl.git",
           devbranch = "main",)
