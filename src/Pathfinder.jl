module Pathfinder

using ADTypes: ADTypes
using Distributions: Distributions
using IrrationalConstants: log2π
using LinearAlgebra
using LogDensityProblems: LogDensityProblems
using OhMyThreads: OhMyThreads
using Optim: Optim, LineSearches
using OptimizationBase: OptimizationBase, OptimizationState
using OptimizationOptimJL: OptimizationOptimJL
using PDMats: PDMats
using ProgressLogging: ProgressLogging
using PSIS: PSIS
using Random
using Requires: Requires
using SciMLBase: SciMLBase
using Statistics: Statistics
using StatsBase: StatsBase

# Declare and export the public API
export PathfinderResult, MultiPathfinderResult
export pathfinder, multipathfinder, resample

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

"""
    _default_turing_chain_type()

Return the default `chain_type` for the Turing extension's `pathfinder`/`multipathfinder`
methods. Methods are defined in package extensions, matching `Turing.sample`'s own default.
"""
function _default_turing_chain_type end

include("utils.jl")
include("woodbury.jl")
include("optimize.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")
include("singlepath.jl")
include("multipath.jl")
include("resample.jl")

function __init__()
    Requires.@require AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d" begin
        include("integration/advancedhmc.jl")
    end
end

end
