using Distributions
using Pathfinder
using Random
using Test

@testset "ELBO estimation" begin
    @testset "maximize_elbo" begin
        target_dist = Normal(0, 0.8)
        logp(x) = logpdf(target_dist, x)
        σs = [1e-3, 0.05, 0.8, 1.0, 1.1, 1.2, 5.0, 10.0]
        dists = Normal.(0, σs)
        rng = MersenneTwister(42)
        lopt, ϕ, logqϕ, λ = Pathfinder.maximize_elbo(rng, logp, dists, 100_000)
        r = σs ./ 0.8
        # explicit elbo calculation
        λexp = (1 .- r.^2) ./ 2 .+ log.(r)
        @test lopt == 3
        @test λ ≈ mean.(logp.(ϕ) .- logqϕ) ≈ λ
        @test λ ≈ λexp rtol=1e-2
        @test λ[lopt] ≈ 0
    end
end
