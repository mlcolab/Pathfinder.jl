using LinearAlgebra
using Distributions
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
            @test length(ϕ) == ndraws
            @test logqϕ ≈ logpdf.(Ref(q), ϕ)
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
        q, _, _ = pathfinder(logp, ∇logp, x0, 10; ndraws_elbo=100)
        @test q.Σ ≈ Σ rtol = 1e-1
    end
end
