using Pathfinder
using Random
using Test
using Transducers

@testset "transducers integration" begin
    @testset "_default_executor" begin
        rng = MersenneTwister(42)
        @test Pathfinder._default_executor(MersenneTwister(42); basesize=2) ===
            SequentialEx()
        if VERSION ≥ v"1.7.0"
            @test Pathfinder._default_executor(Random.GLOBAL_RNG; basesize=1) ===
                PreferParallel(; basesize=1)
            @test Pathfinder._default_executor(Random.default_rng(); basesize=1) ===
                PreferParallel(; basesize=1)
        else
            @test Pathfinder._default_executor(Random.GLOBAL_RNG; basesize=1) ===
                SequentialEx()
            @test Pathfinder._default_executor(Random.default_rng(); basesize=1) ===
                SequentialEx()
        end
    end

    @testset "_findmax" begin
        x = randn(100)
        @test Pathfinder._findmax(x) == findmax(x)
        @test Pathfinder._findmax(x |> Map(sin)) == findmax(sin.(x))
        @test Pathfinder._findmax([NaN, 3.0, 1.0]) === (3.0, 2)
        @test Pathfinder._findmax([NaN, NaN, NaN]) === (NaN, 1)
        @test Pathfinder._findmax([2.0, NaN, 4.0]) === (4.0, 3)
    end
end
