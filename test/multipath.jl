using LinearAlgebra
using Distributions
using ForwardDiff
using Pathfinder
using Test

@testset "multi path pathfinder" begin
    @testset "MvNormal" begin
        n = 10
        nruns = 20
        ndraws = 1000_000
        ndraws_per_run = ndraws ÷ nruns
        Σ = rand_pd_mat(Float64, n)
        μ = randn(n)
        d = MvNormal(μ, Σ)
        logp(x) = logpdf(d, x)
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        x₀s = [rand(Uniform(-2, 2), n) for _ in 1:nruns]
        q, ϕ, component_ids = multipathfinder(
            logp, ∇logp, x₀s, ndraws; ndraws_elbo=100, ndraws_per_run
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
    end
end
