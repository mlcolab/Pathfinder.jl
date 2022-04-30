module Pathfinder

using AbstractDifferentiation: AD
using Distributions: Distributions
using Folds: Folds
# ensure that ForwardDiff is conditionally loaded by GalacticOptim
using ForwardDiff: ForwardDiff
using IrrationalConstants: log2π
using LinearAlgebra
using GalacticOptim: GalacticOptim
using Optim: Optim, LineSearches
using PDMats: PDMats
using ProgressLogging: ProgressLogging
using PSIS: PSIS
using Random
using Requires: Requires
using Statistics: Statistics
using StatsBase: StatsBase
using Transducers: Transducers
using UnPack: @unpack

export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate inverse Hessian positive-definite
const DEFAULT_HISTORY_LENGTH = 6
const DEFAULT_LINE_SEARCH = LineSearches.MoreThuente()
const DEFAULT_OPTIMIZER = Optim.LBFGS(;
    m=DEFAULT_HISTORY_LENGTH, linesearch=DEFAULT_LINE_SEARCH
)
const DEFAULT_NDRAWS_ELBO = 5

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
    Requires.@require DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb" begin
        include("integration/dynamichmc.jl")
    end
    Requires.@require AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d" begin
        include("integration/advancedhmc.jl")
    end
end

end
