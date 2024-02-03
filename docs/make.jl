using PEtab
using Documenter

DocMeta.setdocmeta!(PEtab, :DocTestSetup, :(using PEtab); recursive = true)

makedocs(;
         modules = [PEtab],
         authors = "Viktor Hasselgren, Sebastian Persson, Damiano Ognissanti, Rafael Arutjunjan",
         repo = "https://github.com/sebapersson/PEtab.jl/blob/{commit}{path}#{line}",
         checkdocs = :exports,
         warnonly = false,
         sitename = "PEtab.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://sebapersson.github.io/PEtab.jl",
                                  edit_link = "main",
                                  assets = String[],),
         pages = [
             "Home" => "index.md",
             "Importing problems in PEtab standard format" => "Boehm.md",
             "Defining a PEtab problem in Julia" => Any["Defining parameter estimation problems in Julia" => "Define_in_julia.md",
                                                        "Pre-equilibration (steady-state simulations)" => "Julia_steady_state.md",
                                                        "Noise and observable parameters" => "Julia_obs_noise.md",
                                                        "Condition specific system/model parameters" => "Julia_condition_specific.md",
                                                        "Events (callbacks, dosages etc...)" => "Julia_event.md"],
             "Options for specific problem types" => Any["Models with pre-equilibration (steady-state simulation)" => "Brannmark.md",
                                                         "Medium sized models and adjoint sensitivity analysis" => "Bachmann.md",
                                                         "Models with many conditions specific parameters" => "Beer.md"],
             "Parameter estimation" => Any["Parameter estimation" => "Parameter_estimation.md",
                                           "Available optimisers" => "Avaible_optimisers.md",
                                           "Model selection (PEtab select)" => "Model_selection.md",
                                           "Plots evaluating parameter estimation" => "optimisation_output_plotting.md"],
             "Supported gradient and hessian methods" => "Gradient_hessian_support.md",
             "Choosing the best options for a PEtab problem" => "Best_options.md",
             "API" => "API_choosen.md",
         ],)

deploydocs(;
           repo = "github.com/sebapersson/PEtab.jl.git",
           devbranch = "main",)
