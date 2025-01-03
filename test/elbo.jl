using Distributions
using Pathfinder
using Random
using Test
using Transducers

@testset "ELBO estimation" begin
    @testset "elbo_and_samples!" begin
        σ_target = 0.08
        target_dist = Normal(0, σ_target)
        logp(x) = logpdf(target_dist, x[1])
        σs = [1e-3, 0.05, 0.8, 1.0, 1.1, 1.2, 5.0, 10.0]
        rng = Random.seed!(Random.default_rng(), 42)
        @testset for σ in σs
            dist = Normal(0, σ)
            draws = Matrix{Float64}(undef, length(dist), 1_000_000)
            # rng = Random.seed!(Random.default_rng(), 42)
            est = @inferred Pathfinder.elbo_and_samples!(draws, rng, logp, dist)
            @test est isa Pathfinder.ELBOEstimate

            # explicit elbo calculation
            r = σ / σ_target
            elbo_exp = (1 - r^2) / 2 + log(r)
            @test est.value ≈ elbo_exp atol = 3 * est.std_err
            @test est.draws !== draws
            @test est.log_densities_target ≈ logp.(eachcol(est.draws))
            @test est.log_densities_fit ≈ logpdf.(dist, first.(eachcol(est.draws)))
            @test est.log_density_ratios == est.log_densities_target - est.log_densities_fit
            @test est.value ≈ mean(est.log_density_ratios)
            @test est.std_err ≈ std(est.log_density_ratios) / sqrt(1_000_000)

            # fill!(draws, 0)
            # est_nosave = @inferred Pathfinder.elbo_and_samples!(draws, rng, logp, dist; save_samples=false)
            # @test est_nosave.value == est.value
            # @test est_nosave.std_err == est.std_err
            # @test est_nosave.draws !== draws
            # @test isempty(est_nosave.draws)
        end
    end

    @testset "maximize_elbo" begin
        σ_target = 0.08
        target_dist = Normal(0, σ_target)
        logp(x) = logpdf(target_dist, x[1])
        σs = [1e-3, 0.05, σ_target, 1.0, 1.1, 1.2, 5.0, 10.0]
        dists = Normal.(0, σs)
        executors = [SequentialEx()]
        @testset "$executor" for executor in executors
            rng = Random.seed!(Random.default_rng(), 42)
            lopt, estimates = @inferred Pathfinder.maximize_elbo(
                rng, logp, dists, 100, executor
            )
            @test lopt == 3
            @test estimates[lopt].value ≈ 0
            rng = Random.seed!(Random.default_rng(), 42)
            lopt2, estimates2 = Pathfinder.maximize_elbo(rng, logp, dists, 100, executor)
            @test lopt2 == lopt
            @test getproperty.(estimates2, :value) == getproperty.(estimates, :value)
            @test getproperty.(estimates2, :std_err) == getproperty.(estimates, :std_err)
            lopt3, estimates3 = @inferred Pathfinder.maximize_elbo(
                rng, logp, dists[2:1], 100, executor
            )
            @test lopt3 == 0
            @test isempty(estimates3)
        end
    end
end
