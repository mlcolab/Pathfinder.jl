using ADTypes
using Distributions
using ForwardDiff
using LinearAlgebra
using Pathfinder
using PSIS
using ReverseDiff
using SciMLBase
using Test

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
        rngs = [MersenneTwister(), Random.default_rng()]
        seed = 76
        @testset for rng in rngs, ntasks in unique((1, Threads.nthreads()))
            Random.seed!(rng, seed)
            result = multipathfinder(
                ℓ, ndraws; nruns, ndraws_elbo, ndraws_per_run, rng, ntasks
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
                ℓ, ndraws; nruns, ndraws_elbo, ndraws_per_run, rng, ntasks
            )
            @test result2.fit_distribution == result.fit_distribution
            @test result2.draws == result.draws
            @test result2.draw_component_ids == result.draw_component_ids

            Random.seed!(rng, seed)
            result3 = multipathfinder(
                ℓ, ndraws; nruns, ndraws_elbo, ndraws_per_run, rng, ntasks
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

    @testset "reproducibility across ntasks" begin
        logp(x) = -sum(abs2, x) / 2
        ℓ = build_logdensityproblem(logp, 5, 2)
        seed = 19
        nruns = 6
        ndraws = 200
        nthreads = Threads.nthreads()

        @testset "explicit rng" begin
            rng = MersenneTwister(seed)
            serial = multipathfinder(ℓ, ndraws; rng, nruns, ntasks=1, ntasks_per_run=1)
            Random.seed!(rng, seed)
            threaded = multipathfinder(
                ℓ, ndraws; rng, nruns, ntasks=nthreads, ntasks_per_run=nthreads
            )
            @test serial.draws == threaded.draws
            @test serial.draw_component_ids == threaded.draw_component_ids
            @test [c.μ for c in serial.fit_distribution.components] == [c.μ for c in threaded.fit_distribution.components]
            @test [c.Σ for c in serial.fit_distribution.components] == [c.Σ for c in threaded.fit_distribution.components]
        end

        @testset "default rng" begin
            Random.seed!(seed)
            serial = multipathfinder(ℓ, ndraws; nruns, ntasks=1, ntasks_per_run=1)
            Random.seed!(seed)
            threaded = multipathfinder(
                ℓ, ndraws; nruns, ntasks=nthreads, ntasks_per_run=nthreads
            )
            @test serial.draws == threaded.draws
            @test serial.draw_component_ids == threaded.draw_component_ids
            @test [c.μ for c in serial.fit_distribution.components] == [c.μ for c in threaded.fit_distribution.components]
            @test [c.Σ for c in serial.fit_distribution.components] == [c.Σ for c in threaded.fit_distribution.components]
        end
    end

    @testset "resample(::MultiPathfinderResult)" begin
        dim = 5
        nruns = 4
        ndraws_per_run = 20
        ndraws_total = nruns * ndraws_per_run
        ndraws_new = 8
        logp(x) = -sum(abs2, x) / 2
        ℓ = build_logdensityproblem(logp, dim, 2)
        rng = MersenneTwister(42)
        result = multipathfinder(ℓ, ndraws_per_run; nruns, ndraws_per_run, rng)

        @testset "resample existing draws with replacement" begin
            r2 = resample(result, ndraws_new)
            @test r2 isa MultiPathfinderResult
            @test size(r2.draws) == (dim, ndraws_new)
            @test length(r2.draw_component_ids) == ndraws_new
            @test length(unique(r2.draw_component_ids)) <= nruns
            @test r2.draws_transformed == r2.draws
            @test r2.psis_result === result.psis_result
            draws_all = mapreduce(x -> x.draws, hcat, result.pathfinder_results)
            for col in eachcol(r2.draws)
                @test any(==(col), eachcol(draws_all))
            end
        end

        @testset "resample existing draws without replacement" begin
            r2 = resample(result, ndraws_new; replace=false)
            @test size(r2.draws) == (dim, ndraws_new)
            draws_all = mapreduce(x -> x.draws, hcat, result.pathfinder_results)
            for col in eachcol(r2.draws)
                @test any(==(col), eachcol(draws_all))
            end
            @test length(unique(eachcol(r2.draws))) == ndraws_new
        end

        @testset "resample existing draws without importance" begin
            r2 = resample(result, ndraws_new; importance=false)
            @test r2.psis_result === nothing
            draws_all = mapreduce(x -> x.draws, hcat, result.pathfinder_results)
            for col in eachcol(r2.draws)
                @test any(==(col), eachcol(draws_all))
            end
        end

        @testset "resample existing draws with importance, no stored PSIS" begin
            result_no_psis = multipathfinder(
                ℓ, ndraws_per_run; nruns, ndraws_per_run, rng, importance=false
            )
            @test result_no_psis.psis_result === nothing
            r2 = resample(result_no_psis, ndraws_new)
            @test r2 isa MultiPathfinderResult
            @test r2.psis_result isa PSIS.PSISResult
            draws_all = mapreduce(x -> x.draws, hcat, result_no_psis.pathfinder_results)
            for col in eachcol(r2.draws)
                @test any(==(col), eachcol(draws_all))
            end
        end

        @testset "generate new draws" begin
            r2 = resample(result, ndraws_new; ndraws_per_run=50, replace=true)
            @test r2 isa MultiPathfinderResult
            @test size(r2.draws) == (dim, ndraws_new)
            @test length(r2.draw_component_ids) == ndraws_new
            @test r2.psis_result isa PSIS.PSISResult
            # new draws are NOT necessarily in the original pool
        end

        @testset "generate new draws without importance" begin
            r2 = resample(
                result, ndraws_new; ndraws_per_run=50, importance=false, replace=true
            )
            @test size(r2.draws) == (dim, ndraws_new)
            @test r2.psis_result === nothing
        end

        @testset "non-mutating: original result unchanged" begin
            draws_before = copy(result.draws)
            resample(result, ndraws_new)
            @test result.draws == draws_before
        end

        @testset "preserved fields" begin
            r2 = resample(result, ndraws_new)
            @test r2.input === result.input
            @test r2.optimizer === result.optimizer
            @test r2.fit_distribution === result.fit_distribution
            @test r2.fit_distribution_transformed === result.fit_distribution_transformed
            @test r2.pathfinder_results === result.pathfinder_results
            @test r2.logp === result.logp
        end
    end
end
