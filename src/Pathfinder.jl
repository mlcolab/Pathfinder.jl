module Pathfinder

using Distributions: Distributions
using LinearAlgebra
using Optim: Optim, LineSearches
using PDMats: PDMats
using PSIS: PSIS
using Random
using Statistics: Statistics
using StatsBase: StatsBase
using StatsFuns: log2π

export pathfinder, multipathfinder

# Note: we override the default history length to be shorter and the default line search
# to be More-Thuente, which keeps the approximate inverse Hessian positive-definite
const DEFAULT_OPTIMIZER = Optim.LBFGS(; m=5, linesearch=LineSearches.MoreThuente())

include("woodbury.jl")
include("maximize.jl")
include("inverse_hessian.jl")
include("mvnormal.jl")
include("elbo.jl")
include("resample.jl")
include("singlepath.jl")
include("multipath.jl")

end
