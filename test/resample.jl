using Pathfinder
using PSIS
using Random
using Test

@testset "_resample" begin
    x = randn(100)

    rng = Random.seed!(Random.default_rng(), 42)
    xsub, nothing_result = Pathfinder._resample(rng, x, nothing, 500)
    @test all(in.(xsub, Ref(x)))
    @test nothing_result === nothing

    @testset "replace=false" begin
        xsub, _ = Pathfinder._resample(rng, x, nothing, 10; replace=false)
        @test all(in.(xsub, Ref(x)))
        @test length(unique(xsub)) == 10
    end

    # weight the first 20% much higher than the remaining 80%
    log_weights = randn(100)
    log_weights[1:20] .+= 1_000
    xsub, psis_result = Pathfinder._resample(rng, x, log_weights, 500)
    @test all(in.(xsub, Ref(x[1:20])))
    @test psis_result isa PSIS.PSISResult

    @testset "replace=false with log_weights" begin
        xsub, psis_result = Pathfinder._resample(rng, x, log_weights, 10; replace=false)
        @test all(in.(xsub, Ref(x[1:20])))
        @test psis_result isa PSIS.PSISResult
        @test length(unique(xsub)) == 10
    end

    @testset "from PSISResult" begin
        _, stored_psis = Pathfinder._resample(rng, x, log_weights, 500)
        xsub, psis_result = Pathfinder._resample(rng, x, stored_psis, 10; replace=false)
        @test all(in.(xsub, Ref(x[1:20])))
        @test psis_result === stored_psis
        @test length(unique(xsub)) == 10
    end
end
