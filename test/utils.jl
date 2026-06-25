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

    @testset "_nchunks" begin
        @test Pathfinder._nchunks(0, 4) == 1
        @test Pathfinder._nchunks(1, 4) == 1
        @test Pathfinder._nchunks(10, 1) == 1
        if Threads.nthreads() > 1
            @test Pathfinder._nchunks(10, 4) == 4
            @test Pathfinder._nchunks(2, 4) == 2
        end
    end

    @testset "_maybe_tmap" begin
        xs = randn(50)
        @testset "ntasks=$ntasks" for ntasks in (1, Threads.nthreads())
            @test Pathfinder._maybe_tmap(sin, xs, ntasks) == map(sin, xs)
            @test Pathfinder._maybe_tmap(sin, Float64[], ntasks) == Float64[]
        end
    end

    @testset "_maybe_tmapreduce" begin
        xs = [randn(3) for _ in 1:20]
        @testset "ntasks=$ntasks" for ntasks in (1, Threads.nthreads())
            @test Pathfinder._maybe_tmapreduce(identity, vcat, xs, ntasks) ==
                mapreduce(identity, vcat, xs)
            # `+` on floats is non-associative, so parallel reduction order may differ
            @test Pathfinder._maybe_tmapreduce(sum, +, xs, ntasks) ≈ mapreduce(sum, +, xs)
        end
    end

    @testset "_chunk_tmap" begin
        xs = collect(1:50)
        ys = randn(50)
        @testset "ntasks=$ntasks" for ntasks in (1, Threads.nthreads())
            # matches sequential map, with correct multi-array pairing
            @test Pathfinder._chunk_tmap(xs, ys; ntasks, setup=() -> 0) do _, x, y
                return x + y
            end == map(+, xs, ys)
            # per-chunk state is passed through
            @test Pathfinder._chunk_tmap(xs; ntasks, setup=() -> 10) do s, x
                return s * x
            end == map(x -> 10x, xs)
            # empty arrays return empty
            @test isempty(Pathfinder._chunk_tmap(Int[]; ntasks, setup=() -> 0) do _, x
                return x
            end)
        end

        # the reproducible-seeding idiom yields identical output regardless of ntasks
        function seeded(seed, n, ntasks)
            rng = Random.seed!(Random.default_rng(), seed)
            xs = collect(1:n)
            seeds = rand!(rng, similar(xs, UInt64))
            return Pathfinder._chunk_tmap(xs, seeds; ntasks, setup=() -> copy(rng)) do r, _, s
                Random.seed!(r, s)
                return rand(r)
            end
        end
        @test seeded(7, 50, 1) == seeded(7, 50, Threads.nthreads())
    end
end
