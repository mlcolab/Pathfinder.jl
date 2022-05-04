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
        "Library" => [
            "Public" => "lib/public.md",
            "Internals" => "lib/internals.md",
        ],
        "Examples" => [
            "Quickstart" => "examples/quickstart.md",
            "Initializing HMC" => "examples/initializing-hmc.md",
            "Turing usage" => "examples/turing.md",
        ]
    ],
)

deploydocs(; repo="github.com/sethaxen/Pathfinder.jl", devbranch="main")
