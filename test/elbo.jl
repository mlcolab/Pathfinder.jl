using Distributions
using Pathfinder
using Random
using Test
using Transducers

@testset "ELBO estimation" begin
    @testset "_compute_elbo" begin
        σ_target = 0.08
        target_dist = Normal(0, σ_target)
        logp(x) = logpdf(target_dist, x[1])
        σs = [1e-3, 0.05, 0.8, 1.0, 1.1, 1.2, 5.0, 10.0]
        rng = Random.seed!(Random.default_rng(), 42)
        @testset for σ in σs
            dist = Normal(0, σ)
            est = @inferred Pathfinder._compute_elbo(rng, logp, dist, 1_000_000, Val(true))
            @test est isa Pathfinder.ELBOEstimate

            # explicit elbo calculation
            r = σ / σ_target
            elbo_exp = (1 - r^2) / 2 + log(r)
            @test est.value ≈ elbo_exp atol = 3 * est.std_err
            @test est.log_densities_actual ≈ logp.(eachcol(est.draws))
            @test est.log_densities_fit ≈ logpdf.(dist, first.(eachcol(est.draws)))
            @test est.log_density_ratios == est.log_densities_actual - est.log_densities_fit
            @test est.value ≈ mean(est.log_density_ratios)
            @test est.std_err ≈ std(est.log_density_ratios) / sqrt(1_000_000)
        end
    end

    @testset "MaximumELBO" begin
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
            opt = Pathfinder.MaximumELBO(; rng, executor, ndraws=100, save_draws=true)
            success, dist, lopt, estimates = @inferred opt(logp, nothing, nothing, dists)
            @test lopt == 2
            @test estimates[lopt].value ≈ 0
            Random.seed!(rng, 42)
            success2, dist2, lopt2, estimates2 = @inferred opt(
                logp, nothing, nothing, dists
            )
            @test lopt2 == lopt
            @test getproperty.(estimates2, :value) == getproperty.(estimates, :value)
            @test getproperty.(estimates2, :std_err) == getproperty.(estimates, :std_err)
            success3, dist3, lopt3, estimates3 = @inferred opt(
                logp, nothing, nothing, dists[1:1]
            )
            @test lopt3 == 0
            @test isempty(estimates3)
        end
    end
end
