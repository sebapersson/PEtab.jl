using PEtab
using Documenter

DocMeta.setdocmeta!(PEtab, :DocTestSetup, :(using PEtab); recursive=true)

makedocs(;
    modules=[PEtab],
    authors="Viktor Hasselgren, Sebastian Persson, Damiano Ognissanti, Rafael Arutjunjan",
    repo="https://github.com/sebapersson/PEtab.jl/blob/{commit}{path}#{line}",
    sitename="PEtab.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sebapersson.github.io/PEtab.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "Boehm.md",
        "Tutorials" => Any["Models with pre-equilibration (steady-state simulation)" => "Brannmark.md",
                          "Medium sized models and adjoint sensitivity analysis" => "Bachmann.md",
                          "Models with many conditions specific parameters" => "Beer.md",
                          "Parameter estimation" => "Parameter_estimation.md",
                          "Import Julia ODE model" => "Beer_Julia_import.md"
                          ],
        "Supported gradient and hessian methods" => "Gradient_hessian_support.md",
        "Choosing the best options for a PEtab problem" => "Best_options.md",
        "API" => "API_choosen.md"
    ],
)

deploydocs(;
    repo="github.com/sebapersson/PEtab.jl.git",
    devbranch="main",
)
