using Distributions
using Pathfinder
using Random
using Test
using Transducers

@testset "ELBO estimation" begin
    @testset "elbo_and_samples" begin
        σ_target = 0.08
        target_dist = Normal(0, σ_target)
        logp(x) = logpdf(target_dist, x[1])
        σs = [1e-3, 0.05, 0.8, 1.0, 1.1, 1.2, 5.0, 10.0]
        rng = Random.seed!(Random.default_rng(), 42)
        @testset for σ in σs
            dist = Normal(0, σ)
            elbo, ϕ, logqϕ = @inferred Pathfinder.elbo_and_samples(
                rng, logp, dist, 1_000_000
            )
            # explicit elbo calculation
            r = σ / σ_target
            elbo_exp = (1 - r^2) / 2 + log(r)
            @test elbo ≈ elbo_exp rtol = 1e-2
            @test mean(logp.(eachcol(ϕ)) - logqϕ) ≈ elbo
        end
    end

    @testset "maximize_elbo" begin
        σ_target = 0.08
        target_dist = Normal(0, σ_target)
        logp(x) = logpdf(target_dist, x[1])
        σs = [1e-3, 0.05, σ_target, 1.0, 1.1, 1.2, 5.0, 10.0]
        dists = Normal.(0, σs)
        if VERSION ≥ v"1.7.0"
            executors = [SequentialEx(), ThreadedEx()]
        else
            executors = [SequentialEx()]
        end
        @testset "$executor" for executor in executors
            rng = Random.seed!(Random.default_rng(), 42)
            lopt, elbo, ϕ, logqϕ = @inferred Pathfinder.maximize_elbo(
                rng, logp, dists, 100, executor
            )
            @test lopt == 3
            @test elbo ≈ 0
            rng = Random.seed!(Random.default_rng(), 42)
            lopt2, elbo2, ϕ2, logqϕ2 = Pathfinder.maximize_elbo(
                rng, logp, dists, 100, executor
            )
            @test lopt2 == lopt
            @test elbo2 == elbo
            @test ϕ2 ≈ ϕ
            @test logqϕ ≈ logqϕ2
        end
    end
end
