module Pathfinder

using Distributions: Distributions
using LinearAlgebra
using Optim: Optim, LineSearches
using GalacticOptim: GalacticOptim
using PDMats: PDMats
using PSIS: PSIS
using Random
using Statistics: Statistics
using StatsBase: StatsBase
using StatsFuns: log2π

# ensure that ForwardDiff is conditionally loaded by GalacticOptim
using ForwardDiff: ForwardDiff

export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate inverse Hessian positive-definite
const DEFAULT_HISTORY_LENGTH = 6
const DEFAULT_LINE_SEARCH = LineSearches.MoreThuente()
const DEFAULT_OPTIMIZER = Optim.LBFGS(;
    m=DEFAULT_HISTORY_LENGTH, linesearch=DEFAULT_LINE_SEARCH
)

include("woodbury.jl")
include("maximize.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")
include("resample.jl")
include("singlepath.jl")
include("multipath.jl")

end
