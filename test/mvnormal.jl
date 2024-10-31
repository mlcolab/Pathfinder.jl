using Distributions
using Optim
using Pathfinder
using Random
using SciMLBase
using Test

@testset "MvNormal functions" begin
    @testset "fit_mvnormals" begin
        n = 10
        ℓ = build_logdensityproblem(logp_banana, n, 2)
        θ₀ = 10 * randn(n)
        fun = Pathfinder.build_optim_function(
            ℓ, SciMLBase.NoAD(), LogDensityProblems.capabilities(ℓ)
        )
        prob = SciMLBase.OptimizationProblem(fun, θ₀)
        optimizer = Optim.LBFGS()
        history_length = optimizer.m
        _, optim_trace = Pathfinder.optimize_with_trace(prob, optimizer)
        Σs, num_bfgs_updates_rejected1 = Pathfinder.lbfgs_inverse_hessians(
            optim_trace.points, optim_trace.gradients; history_length
        )
        dists, num_bfgs_updates_rejected2 = @inferred Pathfinder.fit_mvnormals(
            optim_trace.points, optim_trace.gradients; history_length
        )
        @test dists isa Vector{<:MvNormal{Float64,<:Pathfinder.WoodburyPDMat}}
        @test num_bfgs_updates_rejected2 == num_bfgs_updates_rejected1
        @test Σs ≈ getproperty.(dists, :Σ)
        @test optim_trace.points .+ Σs .* optim_trace.gradients ≈ getproperty.(dists, :μ)
    end

    @testset "rand_and_logpdf!" begin
        @testset "MvNormal" begin
            n = 10
            ndraws = 20
            draws = Matrix{Float64}(undef, n, ndraws)
            Σ = rand_pd_mat(Float64, 10)
            μ = randn(n)
            dist = MvNormal(μ, Σ)

            seed = 42
            rng = Random.seed!(Random.default_rng(), seed)
            x, logpx = @inferred Pathfinder.rand_and_logpdf!(rng, dist, draws)
            @test x === draws
            Random.seed!(rng, seed)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf(dist, x2)
            @test x ≈ x2
            @test logpx ≈ logpx2
        end

        @testset "MvNormal{T,Pathfinder.WoodburyPDMat{T}}" begin
            @testset "basic" begin
                n = 10
                ndraws = 20
                nhist = 4
                draws = Matrix{Float64}(undef, n, ndraws)
                A = rand_pd_diag_mat(Float64, n)
                D = rand_pd_mat(Float64, 2nhist)
                B = randn(n, 2nhist)
                Σ = Pathfinder.WoodburyPDMat(A, B, D)
                μ = randn(n)
                dist = MvNormal(μ, Σ)

                seed = 42
                rng = Random.seed!(Random.default_rng(), seed)
                x, logpx = @inferred Pathfinder.rand_and_logpdf!(rng, dist, draws)
                @test x === draws
                Random.seed!(rng, seed)
                x2 = rand(rng, dist, ndraws)
                logpx2 = logpdf(dist, x2)
                @test x ≈ x2
                @test logpx ≈ logpx2
            end

            @testset "consistency of rand" begin
                n = 10
                ndraws = 300_000
                nhist = 4
                A = rand_pd_diag_mat(Float64, n)
                D = rand_pd_mat(Float64, 2nhist)
                B = randn(n, 2nhist)

                Σ = Pathfinder.WoodburyPDMat(A, B, D)
                μ = randn(n)
                dist = MvNormal(μ, Σ)
                v = diag(Σ)
                R = Matrix(Σ) ./ sqrt.(v) ./ sqrt.(v')

                x = rand(dist, ndraws)
                μ_est = mean(x; dims=2)
                v_est = var(x; mean=μ_est, dims=2)
                R_est = cor(x; dims=2)

                nchecks = 2n + div(n * (n - 1), 2)
                α = (0.01 / nchecks) / 2  # multiple correction
                tol = quantile(Normal(), 1 - α) / sqrt(ndraws)

                # asymptotic standard errors for the marginal estimators
                μ_std = sqrt.(v)
                v_std = sqrt(2) * v

                for i in 1:n
                    @test μ_est[i] ≈ μ[i] atol = (tol * μ_std[i])
                    @test v_est[i] ≈ v[i] atol = (tol * v_std[i])
                    for j in (i + 1):n
                        # use variance-stabilizing transformation, recommended in §3.6 of
                        # Van der Vaart, A. W. (2000). Asymptotic statistics (Vol. 3).
                        @test atanh(R_est[i, j]) ≈ atanh(R[i, j]) atol = tol
                    end
                end
            end
        end

        @testset "Normal" begin
            ndraws = 20
            σ = rand() * 10
            μ = randn()
            dist = Normal(μ, σ)

            seed = 42
            rng = Random.seed!(Random.default_rng(), seed)
            draws = Matrix{Float64}(undef, 1, ndraws)
            x, logpx = @inferred Pathfinder.rand_and_logpdf!(rng, dist, draws)
            @test x === draws
            Random.seed!(rng, seed)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf.(dist, x2)
            @test x ≈ x2'
            @test logpx ≈ logpx2
        end
    end
end
