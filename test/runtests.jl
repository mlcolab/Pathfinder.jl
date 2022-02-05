using Pathfinder
using Test

@testset "Pathfinder.jl" begin
    include("woodbury.jl")
    include("optimize.jl")
    include("inverse_hessian.jl")
    include("mvnormal.jl")
    include("elbo.jl")
    include("resample.jl")
    include("singlepath.jl")
    include("multipath.jl")
end
