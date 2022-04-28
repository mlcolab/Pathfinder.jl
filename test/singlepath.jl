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
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 42
        @testset for n in [1, 5, 10, 100], rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            init = randn(n)
            Random.seed!(rng, seed)
            q, ϕ, logqϕ = @inferred pathfinder(logp, ∇logp; init, ndraws, rng, executor)
            @test q isa MvNormal
            @test q.μ ≈ zeros(n)
            @test q.Σ isa Pathfinder.WoodburyPDMat
            @test q.Σ ≈ I
            @test size(q.Σ.B) == (n, 2) # history contains only 1 iteration
            @test ϕ isa AbstractMatrix
            @test size(ϕ) == (n, ndraws)
            @test logqϕ ≈ logpdf(q, ϕ)

            Random.seed!(rng, seed)
            q2, ϕ2, logqϕ2 = pathfinder(logp, ∇logp; init, ndraws, rng, executor)
            @test q2 == q
            @test ϕ2 == ϕ
            @test logqϕ2 == logqϕ

            ndraws = 2
            q3, ϕ3, logqϕ3 = pathfinder(logp, ∇logp; init, ndraws, executor)
            @test size(ϕ3) == (n, ndraws)
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
        dim = 5
        ad_backend = AD.ReverseDiffBackend()
        ndraws_elbo = 100
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 38
        @testset for rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            Random.seed!(rng, seed)
            q, ϕ, logqϕ = @inferred pathfinder(
                logp; rng, dim, ndraws_elbo, ad_backend, executor
            )
            @test q.Σ ≈ Σ rtol = 1e-1
            Random.seed!(rng, seed)
            q2, ϕ2, logqϕ2 = pathfinder(logp; rng, dim, ndraws_elbo, ad_backend, executor)
            @test q2 == q
            @test ϕ2 == ϕ
            @test logqϕ2 == logqϕ
        end
        @testset "kwargs forwarded to solve" begin
            Random.seed!(42)
            i = 0
            cb = (args...,) -> (i += 1; false)
            pathfinder(logp; dim, cb)
            @test i ≠ 6

            Random.seed!(42)
            i = 0
            pathfinder(logp; dim, cb, maxiters=5)
            @test i == 6
        end
    end
    @testset "UniformSampler" begin
        @testset for scale in [1, 2], seed in [42, 38]
            sampler_fun = Pathfinder.UniformSampler(scale)
            rng = Random.seed!(Random.default_rng(), seed)
            x = zeros(100)
            sampler_fun(rng, x)
            @test all(-scale .≤ x .≤ scale)
            Random.seed!(rng, seed)
            x2 = zeros(100)
            x2 .= rand.(rng) .* 2scale .- scale
            @test x2 == x
        end
    end
    @testset "errors if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        init = randn(5)
        prob = GalacticOptim.OptimizationProblem(logp, init, nothing)
        @test_throws ArgumentError pathfinder(prob)
        fun = GalacticOptim.OptimizationFunction(logp, GalacticOptim.AutoForwardDiff())
        prob = GalacticOptim.OptimizationProblem(fun, init, nothing)
        @test_throws ArgumentError pathfinder(prob)
    end
    @testset "errors if neither dim nor init valid" begin
        logp(x) = -sum(abs2, x) / 2
        @test_throws ArgumentError pathfinder(logp)
        @test_throws ArgumentError pathfinder(logp; dim=0)
        pathfinder(logp; dim=3)
        pathfinder(logp; init=randn(5))
    end
end
