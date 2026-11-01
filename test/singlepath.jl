using ADTypes
using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using Pathfinder
using Random
using ReverseDiff
using SciMLBase
using Test

@testset "single path pathfinder" begin
    @testset "IsoNormal" begin
        # here pathfinder finds the exact solution after 1 iteration
        logp(x) = -sum(abs2, x) / 2
        ndraws = 100
        rngs = [MersenneTwister(), Random.default_rng()]
        seed = 42
        @testset for dim in [1, 5, 10, 100],
            rng in rngs,
            ntasks in unique((1, Threads.nthreads()))

            ℓ = build_logdensityproblem(logp, 5, 2)
            init = randn(dim)
            Random.seed!(rng, seed)
            # less restrictive type check to work around https://github.com/mlcolab/Pathfinder.jl/issues/142
            # TODO: remove this workaround once the issue is fixed
            result = @inferred PathfinderResult pathfinder(ℓ; init, ndraws, rng, ntasks)
            @test result isa PathfinderResult
            @test result.input === ℓ
            @test result.optim_prob isa SciMLBase.OptimizationProblem
            @test result.logp(init) ≈ logp(init)
            @test result.rng === rng
            @test result.optimizer ===
                Pathfinder.default_optimizer(Pathfinder.DEFAULT_HISTORY_LENGTH)
            (; fit_distribution) = result
            @test fit_distribution isa MvNormal
            @test fit_distribution.μ ≈ zeros(dim) atol = 1e-6
            @test fit_distribution.Σ isa Pathfinder.WoodburyPDMat
            @test fit_distribution.Σ ≈ I atol = 1e-6
            @test size(fit_distribution.Σ.B) == (dim, 2) # history contains only 1 iteration
            @test result.draws isa AbstractMatrix
            @test size(result.draws) == (dim, ndraws)
            @test result.fit_distribution_transformed === result.fit_distribution
            @test result.draws_transformed === result.draws
            @test result.num_tries ≥ 1
            @test result.optim_solution isa SciMLBase.OptimizationSolution
            @test result.optim_trace isa Pathfinder.OptimizationTrace
            @test result.fit_distributions isa Vector{typeof(fit_distribution)}
            @test length(result.fit_distributions) == length(result.optim_trace)
            @test result.fit_distributions[result.fit_iteration + 1] == fit_distribution
            @test result.fit_iteration ==
                argmax(getproperty.(result.elbo_estimates, :value))

            Random.seed!(rng, seed)
            result2 = pathfinder(ℓ; init, ndraws, rng, ntasks)
            @test result2.fit_iteration == result.fit_iteration
            @test result2.draws == result.draws
            @test getproperty.(result2.elbo_estimates, :value) ==
                getproperty.(result.elbo_estimates, :value)

            ndraws = 2
            result3 = pathfinder(ℓ; init, ndraws, ntasks)
            @test size(result3.draws) == (dim, ndraws)
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
        dim = 5
        ℓ = build_logdensityproblem(logp, dim, 2)
        ndraws_elbo = 500
        rngs = [MersenneTwister(), Random.default_rng()]
        x = randn(dim)
        seed = 38
        optimizer = Optim.LBFGS(; m=6)
        @testset for rng in rngs, ntasks in unique((1, Threads.nthreads()))
            Random.seed!(rng, seed)
            # less restrictive type check to work around https://github.com/mlcolab/Pathfinder.jl/issues/142
            # TODO: remove this workaround once the issue is fixed
            result = @inferred PathfinderResult pathfinder(
                ℓ; rng, optimizer, ndraws_elbo, ntasks
            )
            @test result.input === ℓ
            @test result.fit_distribution.Σ ≈ Σ rtol = 1e-1
            @test result.optimizer == optimizer
            Random.seed!(rng, seed)
            result2 = pathfinder(ℓ; rng, optimizer, ndraws_elbo, ntasks)
            @test result2.fit_distribution == result.fit_distribution
            @test result2.draws == result.draws
            @test getproperty.(result2.elbo_estimates, :value) ==
                getproperty.(result.elbo_estimates, :value)
        end
        @testset "kwargs forwarded to solve" begin
            Random.seed!(42)
            i = 0
            callback = (args...,) -> (i += 1; false)
            pathfinder(ℓ; callback)
            @test i ≠ 4

            Random.seed!(42)
            i = 0
            pathfinder(ℓ; callback, maxiters=3)
            @test i == 4
        end
    end
    @testset "retries" begin
        @testset "logp returning NaN" begin
            dim = 5
            nfail = 20
            logp(x) = i ≤ nfail ? NaN : -sum(abs2, x) / 2
            ℓ = build_logdensityproblem(logp, dim, 2)
            callback = (args...,) -> (i += 1; false)
            i = 1
            result = pathfinder(ℓ; callback)
            @test result.fit_distribution.μ ≈ zeros(dim) atol = 1e-6
            @test result.fit_distribution.Σ ≈ diagm(ones(dim)) atol = 1e-6
            @test result.num_tries == nfail + 1
            @test result.optim_prob.u0 == result.optim_trace.points[1]
            i = 1
            init = randn(dim)
            result2 = pathfinder(ℓ; init, callback, ntries=nfail)
            @test !isapprox(result2.fit_distribution.μ, zeros(dim); atol=1e-6)
            @test result2.fit_iteration == 0
            @test isempty(result2.elbo_estimates)
            @test result2.num_tries == nfail
            @test result2.optim_prob.u0 == result2.optim_trace.points[1]
        end
    end
    @testset "save_trace=false" begin
        # an ill-conditioned Gaussian, so that L-BFGS keeps iterating up to `maxiters`
        dim = 30
        prec = exp10.(range(-2, 2; length=dim))
        logp(x) = -dot(prec, abs2.(x)) / 2
        ℓ = build_logdensityproblem(logp, dim, 1)
        init = ones(dim)
        ndraws_elbo = 4
        same_fit(a, b) = a.μ == b.μ && a.Σ.A == b.Σ.A && a.Σ.B == b.Σ.B && a.Σ.D == b.Σ.D

        @testset for rng in [MersenneTwister(), Random.default_rng()],
            maxiters in [5, 40],
            ndraws in [2, ndraws_elbo, 10]

            seed = 21
            Random.seed!(rng, seed)
            result_trace = pathfinder(ℓ; init, rng, maxiters, ndraws_elbo, ndraws, ntries=1)
            Random.seed!(rng, seed)
            # less restrictive type check to work around https://github.com/mlcolab/Pathfinder.jl/issues/142
            result = @inferred PathfinderResult pathfinder(
                ℓ; init, rng, maxiters, ndraws_elbo, ndraws, ntries=1, save_trace=false
            )
            @test result isa PathfinderResult
            L = length(result_trace.optim_trace) - 1
            @test L == maxiters

            # the same trajectory, stored as log-densities only
            @test length(result.optim_trace) - 1 == L
            @test result.optim_trace.log_densities == result_trace.optim_trace.log_densities
            @test isempty(result.optim_trace.points)
            @test isempty(result.optim_trace.gradients)
            @test result.optim_trace.points isa typeof(result_trace.optim_trace.points)
            @test result.optim_solution.retcode == result_trace.optim_solution.retcode
            @test result.optim_solution.u == result_trace.optim_solution.u
            @test result.num_bfgs_updates_rejected == result_trace.num_bfgs_updates_rejected

            # no fits are kept but the selected one, which is the fit the stored trajectory
            # gives at that iteration
            @test result.fit_distributions isa typeof(result_trace.fit_distributions)
            @test isempty(result.fit_distributions)
            @test 1 ≤ result.fit_iteration ≤ L
            @test same_fit(
                result.fit_distribution,
                result_trace.fit_distributions[result.fit_iteration + 1],
            )
            @test all(isfinite, mean(result.fit_distribution))

            # one ELBO estimate per iteration; only the selected one keeps its draws
            @test result.elbo_estimates isa typeof(result_trace.elbo_estimates)
            @test length(result.elbo_estimates) == L
            values = getproperty.(result.elbo_estimates, :value)
            @test result.fit_iteration == last(Pathfinder._findmax_skipnan(values))
            for (i, est) in enumerate(result.elbo_estimates)
                @test length(est.log_density_ratios) == ndraws_elbo
                @test size(est.draws) == (dim, i == result.fit_iteration ? ndraws_elbo : 0)
            end

            # the returned draws reuse the selected fit's ELBO draws, as with `save_trace`
            @test size(result.draws) == (dim, ndraws)
            k = min(ndraws, ndraws_elbo)
            @test result.draws[:, 1:k] ==
                result.elbo_estimates[result.fit_iteration].draws[:, 1:k]
            @test all(isfinite, result.draws)

            # reproducible
            Random.seed!(rng, seed)
            result2 = pathfinder(
                ℓ; init, rng, maxiters, ndraws_elbo, ndraws, ntries=1, save_trace=false
            )
            @test result2.fit_iteration == result.fit_iteration
            @test result2.draws == result.draws
            @test getproperty.(result2.elbo_estimates, :value) == values
        end

        @testset "memory does not grow with the number of iterations" begin
            # large enough that the O(dim) part of the result dominates its scalars
            dim_large = 2_000
            prec_large = exp10.(range(-2, 2; length=dim_large))
            logp_large(x) = -dot(prec_large, abs2.(x)) / 2
            ℓ_large = build_logdensityproblem(logp_large, dim_large, 1)
            init_large = ones(dim_large)
            rng = MersenneTwister(3)
            sizes = map([20, 80]) do maxiters
                Random.seed!(rng, 3)
                result = pathfinder(
                    ℓ_large;
                    init=init_large,
                    rng,
                    maxiters,
                    ndraws_elbo,
                    ndraws=10,
                    ntries=1,
                    save_trace=false,
                )
                @test length(result.optim_trace) - 1 == maxiters
                return Base.summarysize(result)
            end
            Random.seed!(rng, 3)
            result_trace = pathfinder(
                ℓ_large; init=init_large, rng, maxiters=80, ndraws_elbo, ndraws=10, ntries=1
            )
            # 4x the iterations add only the per-iteration scalars
            @test sizes[2] < 1.05 * sizes[1]
            @test Base.summarysize(result_trace) > 5 * sizes[2]
        end

        @testset "custom callback" begin
            iterations = Int[]
            positions = Vector{Float64}[]
            callback = function (state, args...)
                push!(iterations, state.iter)
                push!(positions, copy(state.u))
                return false
            end
            result = pathfinder(
                ℓ;
                init,
                rng=MersenneTwister(8),
                maxiters=10,
                callback,
                ntries=1,
                save_trace=false,
            )
            @test length(iterations) == length(result.optim_trace) == 11
            @test positions[1] ≈ init
            @test positions[end] ≈ result.optim_solution.u
        end

        @testset "retries" begin
            dim = 5
            nfail = 20
            logp_nan(x) = i ≤ nfail ? NaN : -sum(abs2, x) / 2
            ℓ_nan = build_logdensityproblem(logp_nan, dim, 2)
            callback = (args...,) -> (i += 1; false)
            i = 1
            result = pathfinder(ℓ_nan; callback, save_trace=false)
            @test result.fit_distribution.μ ≈ zeros(dim) atol = 1e-6
            @test result.fit_distribution.Σ ≈ diagm(ones(dim)) atol = 1e-6
            @test result.num_tries == nfail + 1
            @test isempty(result.optim_trace.points)
            i = 1
            init_nan = randn(dim)
            result2 = pathfinder(
                ℓ_nan; init=init_nan, callback, ntries=nfail, save_trace=false
            )
            @test !isapprox(result2.fit_distribution.μ, zeros(dim); atol=1e-6)
            @test result2.fit_iteration == 0
            @test isempty(result2.elbo_estimates)
            @test result2.num_tries == nfail
        end

        @testset "optimizer that reports no gradient" begin
            logp_iso(x) = -sum(abs2, x) / 2
            init_iso = randn(3)
            optimizer = Optim.NelderMead()
            result = pathfinder(
                build_logdensityproblem(logp_iso, 3, 1);
                init=init_iso,
                optimizer,
                maxiters=50,
                save_trace=false,
            )
            @test result.fit_iteration ≥ 1
            @test all(isfinite, result.draws)
            @test_throws ArgumentError pathfinder(
                logp_iso; init=init_iso, optimizer, maxiters=50, save_trace=false
            )
        end
    end

    @testset "UniformSampler" begin
        @test_throws DomainError Pathfinder.UniformSampler(-1.0)
        @test_throws DomainError Pathfinder.UniformSampler(0.0)
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

    @testset "does not error if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        init = randn(5)
        @testset for adtype in [AutoForwardDiff(), AutoReverseDiff()]
            result = pathfinder(logp; init, adtype)
            @test result.optim_prob.f.adtype === adtype
            @test result.fit_distribution.μ ≈ zeros(5) atol = 1e-6
            @test result.fit_distribution.Σ ≈ diagm(ones(5)) atol = 1e-6
        end
    end

    @testset "errors if neither dim nor init valid" begin
        logp(x) = -sum(abs2, x) / 2
        @test_throws ArgumentError pathfinder(logp)
        @test_throws ArgumentError pathfinder(build_logdensityproblem(logp, 0, 2))
        pathfinder(build_logdensityproblem(logp, 3, 2))
    end

    @testset "reproducibility across ntasks" begin
        logp(x) = -sum(abs2, x) / 2
        ℓ = build_logdensityproblem(logp, 5, 2)
        seed = 17

        @testset "explicit rng" begin
            rng = MersenneTwister(seed)
            serial = pathfinder(ℓ; rng, ndraws=100, ntasks=1)
            Random.seed!(rng, seed)
            threaded = pathfinder(ℓ; rng, ndraws=100, ntasks=Threads.nthreads())
            @test serial.draws == threaded.draws
            @test serial.fit_iteration == threaded.fit_iteration
            @test getproperty.(serial.elbo_estimates, :value) ==
                getproperty.(threaded.elbo_estimates, :value)
            @test serial.fit_distribution.μ == threaded.fit_distribution.μ
            @test serial.fit_distribution.Σ == threaded.fit_distribution.Σ
        end

        @testset "default rng" begin
            Random.seed!(seed)
            serial = pathfinder(ℓ; ndraws=100, ntasks=1)
            Random.seed!(seed)
            threaded = pathfinder(ℓ; ndraws=100, ntasks=Threads.nthreads())
            @test serial.draws == threaded.draws
            @test serial.fit_iteration == threaded.fit_iteration
            @test getproperty.(serial.elbo_estimates, :value) ==
                getproperty.(threaded.elbo_estimates, :value)
            @test serial.fit_distribution.μ == threaded.fit_distribution.μ
            @test serial.fit_distribution.Σ == threaded.fit_distribution.Σ
        end
    end
end
