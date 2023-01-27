module Pathfinder

using AbstractDifferentiation: AD
using Distributions: Distributions
using Folds: Folds
# ensure that ForwardDiff is conditionally loaded by Optimization
using ForwardDiff: ForwardDiff
using IrrationalConstants: log2π
using LinearAlgebra
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD
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
using UnPack: @unpack

export PathfinderResult, MultiPathfinderResult
export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate inverse Hessian positive-definite
const DEFAULT_HISTORY_LENGTH = 6
const DEFAULT_LINE_SEARCH = LineSearches.MoreThuente()
const DEFAULT_NDRAWS_ELBO = 5

function default_optimizer(history_length)
    return Optim.LBFGS(; m=history_length, linesearch=DEFAULT_LINE_SEARCH)
end

include("transducers.jl")
include("woodbury.jl")
include("logdensityproblems.jl")
include("optimize.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")
include("resample.jl")
include("singlepath.jl")
include("multipath.jl")

function __init__()
    Requires.@require DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb" begin
        include("integration/dynamichmc.jl")
    end
    Requires.@require AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d" begin
        include("integration/advancedhmc.jl")
    end
    Requires.@require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
        include("integration/turing.jl")
    end
end

end
