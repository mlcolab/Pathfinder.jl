using Pathfinder
using Test

@testset "Pathfinder.jl" begin
    include("woodbury.jl")
    include("inverse_hessian.jl")
end
