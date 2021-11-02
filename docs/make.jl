using Pathfinder
using Documenter

DocMeta.setdocmeta!(Pathfinder, :DocTestSetup, :(using Pathfinder); recursive=true)

makedocs(;
    modules=[Pathfinder],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo="https://github.com/sethaxen/Pathfinder.jl/blob/{commit}{path}#{line}",
    sitename="Pathfinder.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sethaxen.github.io/Pathfinder.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Single-path Pathfinder" => "pathfinder.md",
        "Multi-path Pathfinder" => "multipathfinder.md",
        "Using Pathfinder with Turing" => "turing.md",
    ],
)

deploydocs(; repo="github.com/sethaxen/Pathfinder.jl", devbranch="main")
