using ADTypes
using Distributions
using ForwardDiff
using LinearAlgebra
using Optimization
using Pathfinder
using PSIS
using ReverseDiff
using SciMLBase
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
        ℓ = build_logdensityproblem(logp, dim, 2)
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 76
        @testset for rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()

            Random.seed!(rng, seed)
            result = multipathfinder(
                ℓ, ndraws; nruns, ndraws_elbo, ndraws_per_run, rng, executor
            )
            @test result isa MultiPathfinderResult
            @test result.input === ℓ
            @test result.optim_fun isa SciMLBase.OptimizationFunction
            @test result.rng === rng
            @test result.optimizer ===
                Pathfinder.default_optimizer(Pathfinder.DEFAULT_HISTORY_LENGTH)
            @test result.fit_distribution isa MixtureModel
            @test ncomponents(result.fit_distribution) == nruns
            @test Distributions.component_type(result.fit_distribution) <: MvNormal
            @test result.draws isa AbstractMatrix
            @test size(result.draws) == (dim, ndraws)
            @test result.draw_component_ids isa Vector{Int}
            @test length(result.draw_component_ids) == ndraws
            @test extrema(result.draw_component_ids) == (1, nruns)
            @test result.fit_distribution_transformed === result.fit_distribution
            @test result.draws_transformed == result.draws
            @test result.pathfinder_results isa Vector{<:PathfinderResult}
            @test length(result.pathfinder_results) == nruns
            @test result.psis_result isa PSIS.PSISResult
            μ_hat = mean(result.draws; dims=2)
            Σ_hat = cov(result.draws .- μ_hat; dims=2, corrected=false)
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
            result2 = multipathfinder(
                ℓ, ndraws; nruns, ndraws_elbo, ndraws_per_run, rng, executor
            )
            @test result2.fit_distribution == result.fit_distribution
            @test result2.draws == result.draws
            @test result2.draw_component_ids == result.draw_component_ids

            Random.seed!(rng, seed)
            result3 = multipathfinder(
                ℓ, ndraws; nruns, ndraws_elbo, ndraws_per_run, rng, executor
            )
            for (c1, c2) in
                zip(result.fit_distribution.components, result3.fit_distribution.components)
                @test c1 ≈ c2 atol = 1e-6
            end
        end

        init = [randn(dim) for _ in 1:nruns]
        result = multipathfinder(ℓ, ndraws; init)
        @test ncomponents(result.fit_distribution) == nruns
        @test size(result.draws) == (dim, ndraws)
    end

    @testset "does not error if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        init = [randn(5) for _ in 1:10]
        @testset for adtype in [AutoForwardDiff(), AutoReverseDiff()]
            result = multipathfinder(logp, 10; init, adtype)
            @test result.optim_fun.adtype === adtype
            for component in result.fit_distribution.components
                @test component.μ ≈ zeros(5) atol = 1e-6
                @test component.Σ ≈ I(5) atol = 1e-6
            end
        end
    end

    @testset "errors if neither init nor nruns valid" begin
        logp(x) = -sum(abs2, x) / 2
        ℓ = build_logdensityproblem(logp, 5, 2)
        @test_throws ArgumentError multipathfinder(ℓ, 10; nruns=0)
        multipathfinder(ℓ, 10; nruns=2)
    end
end
