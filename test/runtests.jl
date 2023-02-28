using Pathfinder
using Random
using Test

Random.seed!(0)

@testset "Pathfinder.jl" begin
    include("transducers.jl")
    include("woodbury.jl")
    include("callbacks.jl")
    include("optimize.jl")
    include("inverse_hessian.jl")
    include("mvnormal.jl")
    include("elbo.jl")
    include("resample.jl")
    include("singlepath.jl")
    include("multipath.jl")
end
