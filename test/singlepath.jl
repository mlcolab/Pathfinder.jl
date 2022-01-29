using LinearAlgebra
using Distributions
using ForwardDiff
using GalacticOptim
using Pathfinder
using Test

@testset "single path pathfinder" begin
    @testset "IsoNormal" begin
        # here pathfinder finds the exact solution after 1 iteration
        logp(x) = -sum(abs2, x) / 2
        ∇logp(x) = -x
        ndraws = 100
        @testset for n in [1, 5, 10, 100]
            x0 = randn(n)
            q, ϕ, logqϕ = @inferred pathfinder(logp, ∇logp, x0, ndraws)
            @test q isa MvNormal
            @test q.μ ≈ zeros(n)
            @test q.Σ isa Pathfinder.WoodburyPDMat
            @test q.Σ ≈ I
            @test size(q.Σ.B) == (n, 2) # history contains only 1 iteration
            @test ϕ isa AbstractMatrix
            @test size(ϕ) == (n, ndraws)
            @test logqϕ ≈ logpdf(q, ϕ)

            q2, ϕ2, logqϕ2 = pathfinder(logp, ∇logp, x0, 2)
            @test size(ϕ2) == (n, 2)
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
        rng = MersenneTwister(38)
        q, _, _ = pathfinder(logp, ∇logp, x₀, 10; rng=rng, ndraws_elbo=100)
        @test q.Σ ≈ Σ rtol = 1e-1
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
