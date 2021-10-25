using Distributions
using Pathfinder
using Random
using Test

include("test_utils.jl")

@testset "MvNormal functions" begin
    @testset "rand_and_logpdf" begin
        ndraws = 20
        @testset "MvNormal" begin
            n = 10
            Σ = rand_pd_mat(Float64, 10)
            μ = randn(n)
            dist = MvNormal(μ, Σ)

            rng = MersenneTwister(42)
            x, logpx = Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            rng = MersenneTwister(42)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf(dist, x2)
            @test x ≈ collect(eachcol(x2))
            @test logpx ≈ logpx2
        end

        @testset "Normal" begin
            σ = rand() * 10
            μ = randn()
            dist = Normal(μ, σ)

            rng = MersenneTwister(42)
            x, logpx = Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            rng = MersenneTwister(42)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf.(dist, x2)
            @test x ≈ x2
            @test logpx ≈ logpx2
        end
    end
end
