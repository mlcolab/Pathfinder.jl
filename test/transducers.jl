using Pathfinder
using Random
using Test
using Transducers

@testset "transducers integration" begin
    @testset "_findmax" begin
        x = randn(100)
        @test Pathfinder._findmax(x) == findmax(x)
        @test Pathfinder._findmax(x |> Map(sin)) == findmax(sin.(x))
        @test Pathfinder._findmax([NaN, 3.0, 1.0]) === (3.0, 2)
        @test Pathfinder._findmax([NaN, NaN, NaN]) === (NaN, 1)
        @test Pathfinder._findmax([2.0, NaN, 4.0]) === (4.0, 3)
    end
end
