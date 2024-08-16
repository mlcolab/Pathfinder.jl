using Pathfinder
using Documenter

DocMeta.setdocmeta!(Pathfinder, :DocTestSetup, :(using Pathfinder); recursive=true)

makedocs(;
    modules=[Pathfinder],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo=Remotes.GitHub("mlcolab", "Pathfinder.jl"),
    sitename="Pathfinder.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mlcolab.github.io/Pathfinder.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Library" => ["Public" => "lib/public.md", "Internals" => "lib/internals.md"],
        "Examples" => [
            "Quickstart" => "examples/quickstart.md",
            "Initializing HMC" => "examples/initializing-hmc.md",
            "Turing usage" => "examples/turing.md",
        ],
    ],
)

if get(ENV, "DEPLOY_DOCS", "true") == "true"
    deploydocs(;
        repo="github.com/mlcolab/Pathfinder.jl", devbranch="main", push_preview=true
    )
end
