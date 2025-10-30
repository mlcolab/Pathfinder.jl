using Pathfinder
using AdvancedHMC  # load RankEuclideanMetric using Requires
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using DynamicPPL # bring type names into scope for @docs signatures
using Turing # load extension

# Expose extension module in Main so @docs blocks can reference it
const PathfinderTuringExt = Base.get_extension(Pathfinder, :PathfinderTuringExt)

DocMeta.setdocmeta!(Pathfinder, :DocTestSetup, :(using Pathfinder); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:numeric)

links = InterLinks(
    "AdvancedHMC" => "https://turinglang.org/AdvancedHMC.jl/stable/",
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "Distributions" => (
        "https://juliastats.org/Distributions.jl/stable/",
        "https://juliastats.org/Distributions.jl/dev/objects.inv",
        joinpath(@__DIR__, "inventories", "Distributions.toml"),
    ),
    "DynamicHMC" => (
        "https://www.tamaspapp.eu/DynamicHMC.jl/stable/",
        "https://www.tamaspapp.eu/DynamicHMC.jl/dev/objects.inv",
        joinpath(@__DIR__, "inventories", "DynamicHMC.toml"),
    ),
    "DynamicPPL" => "https://turinglang.org/DynamicPPL.jl/stable/",
    "LogDensityProblems" => "https://www.tamaspapp.eu/LogDensityProblems.jl/stable/",
    "MCMCChains" => (
        "https://turinglang.org/MCMCChains.jl/stable/",
        "https://turinglang.org/MCMCChains.jl/dev/objects.inv",
    ),
    "Optim" => "https://julianlsolvers.github.io/Optim.jl/stable/",
    "Optimization" => "https://docs.sciml.ai/Optimization/stable/",
    "PSIS" => "https://julia.arviz.org/PSIS/stable/",
    "Transducers" => (
        "https://juliafolds2.github.io/Transducers.jl/stable/",  # not built for a while
        "https://juliafolds2.github.io/Transducers.jl/dev/objects.inv",
        joinpath(@__DIR__, "inventories", "Transducers.toml"),
    ),
)

makedocs(;
    modules=[
        Pathfinder,
        Base.get_extension(Pathfinder, :PathfinderTuringExt),
    ],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo=Remotes.GitHub("mlcolab", "Pathfinder.jl"),
    sitename="Pathfinder.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mlcolab.github.io/Pathfinder.jl",
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Library" => ["Public" => "lib/public.md", "Internals" => "lib/internals.md"],
        "Examples" => [
            "Quickstart" => "examples/quickstart.md",
            "Initializing HMC" => "examples/initializing-hmc.md",
            "Turing usage" => "examples/turing.md",
        ],
        "References" => "references.md",
    ],
    plugins=[bib, links],
    warnonly=[:missing_docs],
)

if get(ENV, "DEPLOY_DOCS", "true") == "true"
    deploydocs(;
        repo="github.com/mlcolab/Pathfinder.jl", devbranch="main", push_preview=true
    )
end
