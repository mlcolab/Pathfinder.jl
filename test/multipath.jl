using AbstractDifferentiation
using Distributions
using ForwardDiff
using LinearAlgebra
using GalacticOptim
using Pathfinder
using ReverseDiff
using Test
using Transducers

@testset "multi path pathfinder" begin
    @testset "MvNormal" begin
        dim = 10
        nruns = 20
        ndraws = 1000_000
        ndraws_per_run = ndraws ÷ nruns
        ndraws_elbo = 100
        Σ = rand_pd_mat(Float64, dim)
        μ = randn(dim)
        d = MvNormal(μ, Σ)
        logp(x) = logpdf(d, x)
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 76
        @testset for rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            Random.seed!(rng, seed)
            q, ϕ, component_ids = multipathfinder(
                logp, ∇logp, ndraws; dim, nruns, ndraws_elbo, ndraws_per_run, rng, executor
            )
            @test q isa MixtureModel
            @test ncomponents(q) == nruns
            @test Distributions.component_type(q) <: MvNormal
            @test ϕ isa AbstractMatrix
            @test size(ϕ) == (dim, ndraws)
            @test component_ids isa Vector{Int}
            @test length(component_ids) == ndraws
            @test extrema(component_ids) == (1, nruns)
            μ_hat = mean(ϕ; dims=2)
            Σ_hat = cov(ϕ .- μ_hat; dims=2, corrected=false)
            # adapted from the MvNormal tests
            # allow for 15x disagreement in atol, since this method is approximate
            multiplier = 15
            for i in eachindex(μ)
                atol = sqrt(Σ[i, i] / ndraws) * 8 * multiplier
                @test μ_hat[i] ≈ μ[i] atol = atol
            end
            for i in axes(Σ, 1), j in axes(Σ, 2)
                atol = sqrt(Σ[i, i] * Σ[j, j] / ndraws) * 10 * multiplier
                @test isapprox(Σ_hat[i, j], Σ[i, j], atol=atol)
            end

            Random.seed!(rng, seed)
            q2, ϕ2, component_ids2 = multipathfinder(
                logp, ndraws; dim, nruns, ndraws_elbo, ndraws_per_run, rng, executor
            )
            @test q2 == q
            @test ϕ2 == ϕ
            @test component_ids2 == component_ids

            Random.seed!(rng, seed)
            ad_backend = AD.ReverseDiffBackend()
            q3, ϕ3, component_ids3 = multipathfinder(
                logp,
                ndraws;
                dim,
                nruns,
                ndraws_elbo,
                ndraws_per_run,
                rng,
                executor,
                ad_backend,
            )
            for (c1, c2) in zip(q.components, q3.components)
                @test c1 ≈ c2 atol = 1e-6
            end
        end

        init = [randn(dim) for _ in 1:nruns]
        q, ϕ, component_ids = multipathfinder(logp, ∇logp, ndraws; init)
        @test ncomponents(q) == nruns
        @test size(ϕ) == (dim, ndraws)
    end
    @testset "errors if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        init = [randn(5) for _ in 1:10]
        fun = GalacticOptim.OptimizationFunction(logp, GalacticOptim.AutoForwardDiff())
        @test_throws ArgumentError multipathfinder(fun, 10; init)
    end
    @testset "errors if neither dim nor init valid" begin
        logp(x) = -sum(abs2, x) / 2
        nruns = 2
        @test_throws ArgumentError multipathfinder(logp, 10; nruns)
        @test_throws ArgumentError multipathfinder(logp, 10; nruns, dim=0)
        multipathfinder(logp, 10; nruns, dim=2)
        multipathfinder(logp, 10; init=[randn(2) for _ in 1:nruns])
    end
    @testset "errors if neither init nor nruns valid" begin
        logp(x) = -sum(abs2, x) / 2
        @test_throws ArgumentError multipathfinder(logp, 10; dim=5, nruns=0)
        multipathfinder(logp, 10; dim=5, nruns=2)
    end
end
