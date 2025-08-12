module Pathfinder

using ADTypes: ADTypes
using Distributions: Distributions
using Folds: Folds
using IrrationalConstants: log2π
using LinearAlgebra
using LogDensityProblems: LogDensityProblems
using Optim: Optim, LineSearches
using Optimization: Optimization
using OptimizationOptimJL: OptimizationOptimJL
using PDMats: PDMats
using ProgressLogging: ProgressLogging
using PSIS: PSIS
using Random
using Requires: Requires
using SciMLBase: SciMLBase
using Statistics: Statistics
using StatsBase: StatsBase
using Transducers: Transducers

# Declare and export the public API
export PathfinderResult, MultiPathfinderResult
export pathfinder, multipathfinder

const DEFAULT_HISTORY_LENGTH = 6
const DEFAULT_LINE_SEARCH = LineSearches.HagerZhang()
const DEFAULT_LINE_SEARCH_INIT = LineSearches.InitialHagerZhang()
const DEFAULT_NDRAWS_ELBO = 5

function default_optimizer(history_length)
    return Optim.LBFGS(;
        m=history_length,
        linesearch=DEFAULT_LINE_SEARCH,
        alphaguess=DEFAULT_LINE_SEARCH_INIT,
    )
end

# We depend on Optim, and Optim depends on ForwardDiff, so we can offer it as a default.
default_ad() = ADTypes.AutoForwardDiff()

include("transducers.jl")
include("woodbury.jl")
include("optimize.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")
include("resample.jl")
include("singlepath.jl")
include("multipath.jl")

function __init__()
    Requires.@require AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d" begin
        include("integration/advancedhmc.jl")
    end
end

end
