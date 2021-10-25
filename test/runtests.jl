using Pathfinder
using Test

@testset "Pathfinder.jl" begin
    include("woodbury.jl")
    include("inverse_hessian.jl")
    include("mvnormal.jl")
    include("elbo.jl")
end
