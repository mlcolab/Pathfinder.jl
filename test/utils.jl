using Pathfinder
using Random
using Test

@testset "parallel helpers" begin
    @testset "_findmax_skipnan" begin
        x = randn(100)
        @test Pathfinder._findmax_skipnan(x) == findmax(x)
        @test Pathfinder._findmax_skipnan(sin, x) == findmax(sin.(x))
        @test Pathfinder._findmax_skipnan([NaN, 3.0, 1.0]) === (3.0, 2)
        @test Pathfinder._findmax_skipnan([NaN, NaN, NaN]) === (NaN, 1)
        @test Pathfinder._findmax_skipnan([2.0, NaN, 4.0]) === (4.0, 3)
    end
end
