using PEtab
using Documenter

DocMeta.setdocmeta!(PEtab, :DocTestSetup, :(using PEtab); recursive=true)

makedocs(;
    modules=[PEtab],
    authors="Viktor Hasselgren, Sebastian Persson, Rafael Arutjunjan",
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
    ],
)

deploydocs(;
    repo="github.com/sebapersson/PEtab.jl",
    devbranch="main",
)
