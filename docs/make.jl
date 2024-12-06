using Pathfinder
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(Pathfinder, :DocTestSetup, :(using Pathfinder); recursive=true)

links = InterLinks(
    "AdvancedHMC" => "https://turinglang.org/AdvancedHMC.jl/stable/",
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "DynamicPPL" => "https://turinglang.org/DynamicPPL.jl/stable/",
    "LogDensityProblems" => "https://www.tamaspapp.eu/LogDensityProblems.jl/stable/",
    "MCMCChains" => (
        "https://turinglang.org/MCMCChains.jl/stable/",
        "https://turinglang.org/MCMCChains.jl/dev/objects.inv",
    ),
    "Optim" => "https://julianlsolvers.github.io/Optim.jl/stable/",
    "Optimization" => "https://docs.sciml.ai/Optimization/stable/",
    "PSIS" => "https://julia.arviz.org/PSIS/stable/",
)

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
    plugins=[links],
)

if get(ENV, "DEPLOY_DOCS", "true") == "true"
    deploydocs(;
        repo="github.com/mlcolab/Pathfinder.jl", devbranch="main", push_preview=true
    )
end
