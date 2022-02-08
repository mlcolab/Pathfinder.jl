module Pathfinder

using AbstractDifferentiation: AD
using Distributions: Distributions
# ensure that ForwardDiff is conditionally loaded by GalacticOptim
using ForwardDiff: ForwardDiff
using LinearAlgebra
using GalacticOptim: GalacticOptim
using Optim: Optim, LineSearches
using PDMats: PDMats
using PSIS: PSIS
using Random
using Setfield: Setfield
using Statistics: Statistics
using StatsBase: StatsBase
using StatsFuns: log2π

export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate inverse Hessian positive-definite
const DEFAULT_HISTORY_LENGTH = 6
const DEFAULT_LINE_SEARCH = LineSearches.MoreThuente()
const DEFAULT_OPTIMIZER = Optim.LBFGS(;
    m=DEFAULT_HISTORY_LENGTH, linesearch=DEFAULT_LINE_SEARCH
)

include("woodbury.jl")
include("optimize.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")
include("resample.jl")
include("singlepath.jl")
include("multipath.jl")

end
