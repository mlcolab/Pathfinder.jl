using Pathfinder
using Random
using Test

Random.seed!(0)

@testset "Pathfinder.jl" begin
    include("test_utils.jl")
    include("transducers.jl")
    include("woodbury.jl")
    include("optimize.jl")
    include("lbfgs.jl")
    include("mvnormal.jl")
    include("elbo.jl")
    include("resample.jl")
    include("singlepath.jl")
    include("multipath.jl")
end
