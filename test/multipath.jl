using AbstractDifferentiation
using Distributions
using ForwardDiff
using LinearAlgebra
using GalacticOptim
using Pathfinder
using ReverseDiff
using Test

@testset "multi path pathfinder" begin
    @testset "MvNormal" begin
        rng = MersenneTwister(42)
        n = 10
        nruns = 20
        ndraws = 1000_000
        ndraws_per_run = ndraws ÷ nruns
        ndraws_elbo = 100
        Σ = rand_pd_mat(rng, Float64, n)
        μ = randn(rng, n)
        d = MvNormal(μ, Σ)
        logp(x) = logpdf(d, x)
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        x₀s = [rand(rng, Uniform(-2, 2), n) for _ in 1:nruns]
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        @testset for rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            Random.seed!(rng, 76)
            q, ϕ, component_ids = multipathfinder(
                logp, ∇logp, x₀s, ndraws; ndraws_elbo, ndraws_per_run, rng, executor
            )
            @test q isa MixtureModel
            @test ncomponents(q) == nruns
            @test Distributions.component_type(q) <: MvNormal
            @test ϕ isa AbstractMatrix
            @test size(ϕ) == (n, ndraws)
            @test component_ids isa Vector{Int}
            @test length(component_ids) == ndraws
            @test extrema(component_ids) == (1, nruns)
            μ_hat = mean(ϕ; dims=2)
            Σ_hat = cov(ϕ .- μ_hat; dims=2, corrected=false)
            # adapted from the MvNormal tests
            # allow for 15x disagreement in atol, since this method is approximate
            multiplier = 15
            for i in 1:n
                atol = sqrt(Σ[i, i] / ndraws) * 8 * multiplier
                @test μ_hat[i] ≈ μ[i] atol = atol
            end
            for i in 1:n, j in 1:n
                atol = sqrt(Σ[i, i] * Σ[j, j] / ndraws) * 10 * multiplier
                @test isapprox(Σ_hat[i, j], Σ[i, j], atol=atol)
            end

            Random.seed!(rng, 76)
            q2, ϕ2, component_ids2 = multipathfinder(
                logp, x₀s, ndraws; ndraws_elbo, ndraws_per_run, rng, executor
            )
            @test q2 == q
            @test ϕ2 == ϕ
            @test component_ids2 == component_ids

            Random.seed!(rng, 76)
            ad_backend = AD.ReverseDiffBackend()
            q3, ϕ3, component_ids3 = multipathfinder(
                logp, x₀s, ndraws; ndraws_elbo, ndraws_per_run, rng, executor, ad_backend
            )
            for (c1, c2) in zip(q.components, q3.components)
                @test c1 ≈ c2 atol = 1e-6
            end
        end
    end
    @testset "errors if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        x0s = [randn(5) for _ in 1:10]
        fun = GalacticOptim.OptimizationFunction(logp, GalacticOptim.AutoForwardDiff())
        @test_throws ArgumentError multipathfinder(fun, x0s, 10)
    end
end
