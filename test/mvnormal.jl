using AbstractDifferentiation
using Distributions
using ForwardDiff
using Optim
using Pathfinder
using Random
using Test

include("test_utils.jl")

@testset "MvNormal functions" begin
    @testset "fit_mvnormals" begin
        n = 10
        logp(x) = logp_banana(x)
        θ₀ = 10 * randn(n)
        ad_backend = AD.ForwardDiffBackend()
        fun = Pathfinder.build_optim_function(logp; ad_backend)
        prob = Pathfinder.build_optim_problem(fun, θ₀)
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

    @testset "rand_and_logpdf" begin
        ndraws = 20
        @testset "MvNormal" begin
            n = 10
            Σ = rand_pd_mat(Float64, 10)
            μ = randn(n)
            dist = MvNormal(μ, Σ)

            seed = 42
            rng = Random.seed!(Random.default_rng(), seed)
            x, logpx = @inferred Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            Random.seed!(rng, seed)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf(dist, x2)
            @test x ≈ x2
            @test logpx ≈ logpx2
        end

        @testset "MvNormal{T,Pathfinder.WoodburyPDMat{T}}" begin
            n = 10
            ndraws = 20
            nhist = 4
            A = rand_pd_diag_mat(Float64, 10)
            D = rand_pd_mat(Float64, 2nhist)
            B = randn(n, 2nhist)
            Σ = Pathfinder.WoodburyPDMat(A, B, D)
            μ = randn(n)
            dist = MvNormal(μ, Σ)

            seed = 42
            rng = Random.seed!(Random.default_rng(), seed)
            x, logpx = @inferred Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            Random.seed!(rng, seed)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf(dist, x2)
            @test x ≈ x2
            @test logpx ≈ logpx2
        end

        @testset "Normal" begin
            σ = rand() * 10
            μ = randn()
            dist = Normal(μ, σ)

            seed = 42
            rng = Random.seed!(Random.default_rng(), seed)
            x, logpx = @inferred Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            Random.seed!(rng, seed)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf.(dist, x2)
            @test x ≈ x2'
            @test logpx ≈ logpx2
        end
    end
end
