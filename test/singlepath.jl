using AbstractDifferentiation
using Distributions
using ForwardDiff
using GalacticOptim
using LinearAlgebra
using Pathfinder
using Random
using ReverseDiff
using Test
using Transducers

@testset "single path pathfinder" begin
    @testset "IsoNormal" begin
        # here pathfinder finds the exact solution after 1 iteration
        logp(x) = -sum(abs2, x) / 2
        ∇logp(x) = -x
        ndraws = 100
        @testset for n in [1, 5, 10, 100], rng in [MersenneTwister(), Random.default_rng()]
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            x0 = randn(n)
            Random.seed!(rng, 42)
            q, ϕ, logqϕ = @inferred pathfinder(logp, ∇logp, x0, ndraws; rng, executor)
            @test q isa MvNormal
            @test q.μ ≈ zeros(n)
            @test q.Σ isa Pathfinder.WoodburyPDMat
            @test q.Σ ≈ I
            @test size(q.Σ.B) == (n, 2) # history contains only 1 iteration
            @test ϕ isa AbstractMatrix
            @test size(ϕ) == (n, ndraws)
            @test logqϕ ≈ logpdf(q, ϕ)

            Random.seed!(rng, 42)
            q2, ϕ2, logqϕ2 = pathfinder(logp, ∇logp, x0, ndraws; rng, executor)
            @test q2 == q
            @test ϕ2 == ϕ
            @test logqϕ2 == logqϕ

            q3, ϕ3, logqϕ3 = pathfinder(logp, ∇logp, x0, 2; executor)
            @test size(ϕ3) == (n, 2)
        end
    end
    @testset "MvNormal" begin
        #! format: off
        Σ = [
            2.71   0.5    0.19   0.07   1.04
            0.5    1.11  -0.08  -0.17  -0.08
            0.19  -0.08   0.26   0.07  -0.7
            0.07  -0.17   0.07   0.11  -0.21
            1.04  -0.08  -0.7   -0.21   8.65
        ]
        #! format: on
        P = inv(Symmetric(Σ))
        logp(x) = -dot(x, P, x) / 2
        ∇logp(x) = -(P * x)
        x₀ = [2.08, 3.77, -1.26, -0.97, -3.91]
        ad_backend = AD.ReverseDiffBackend()
        ndraws_elbo = 100
        @testset for rng in [MersenneTwister(), Random.default_rng()]
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            Random.seed!(rng, 38)
            q, ϕ, logqϕ = @inferred pathfinder(
                logp, x₀, 10; rng, ndraws_elbo, ad_backend, executor
            )
            @test q.Σ ≈ Σ rtol = 1e-1
            Random.seed!(rng, 38)
            q2, ϕ2, logqϕ2 = pathfinder(
                logp, x₀, 10; rng, ndraws_elbo, ad_backend, executor
            )
            @test q2 == q
            @test ϕ2 == ϕ
            @test logqϕ2 == logqϕ
        end
    end
    @testset "errors if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        x0 = randn(5)
        prob = GalacticOptim.OptimizationProblem(logp, x0, nothing)
        @test_throws ArgumentError pathfinder(prob, 10)
        fun = GalacticOptim.OptimizationFunction(logp, GalacticOptim.AutoForwardDiff())
        prob = GalacticOptim.OptimizationProblem(fun, x0, nothing)
        @test_throws ArgumentError pathfinder(prob, 10)
    end
end
