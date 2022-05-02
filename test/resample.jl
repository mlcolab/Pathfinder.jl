using Pathfinder
using PSIS
using Random
using Test

@testset "resample" begin
    x = randn(100)

    rng = Random.seed!(Random.default_rng(), 42)
    xsub = Pathfinder.resample(rng, x, 500)
    @test all(in.(xsub, Ref(x)))

    # weight the first 20% much higher than the remaining 80%
    log_weights = randn(100)
    log_weights[1:20] .+= 1_000
    xsub, result = Pathfinder.resample(rng, x, log_weights, 500)
    @test all(in.(xsub, Ref(x[1:20])))
    @test result isa PSIS.PSISResult
end
