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
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        θ₀ = 10 * randn(n)
        optimizer = Optim.LBFGS()
        θs, logpθs, ∇logpθs = Pathfinder.maximize_with_trace(logp, ∇logp, θ₀, optimizer)
        Σs = Pathfinder.lbfgs_inverse_hessians(θs, ∇logpθs; history_length=optimizer.m)
        dists = @inferred Pathfinder.fit_mvnormals(θs, ∇logpθs; history_length=optimizer.m)
        @test dists isa Vector{<:MvNormal{Float64,<:Pathfinder.WoodburyPDMat}}
        @test Σs ≈ getproperty.(dists, :Σ)
        @test θs .+ Σs .* ∇logpθs ≈ getproperty.(dists, :μ)
    end

    @testset "rand_and_logpdf" begin
        ndraws = 20
        @testset "MvNormal" begin
            n = 10
            Σ = rand_pd_mat(Float64, 10)
            μ = randn(n)
            dist = MvNormal(μ, Σ)

            rng = MersenneTwister(42)
            x, logpx = @inferred Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            rng = MersenneTwister(42)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf(dist, x2)
            @test x ≈ collect(eachcol(x2))
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

            rng = MersenneTwister(42)
            x, logpx = @inferred Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            rng = MersenneTwister(42)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf(dist, x2)
            @test x ≈ collect(eachcol(x2))
            @test logpx ≈ logpx2
        end

        @testset "Normal" begin
            σ = rand() * 10
            μ = randn()
            dist = Normal(μ, σ)

            rng = MersenneTwister(42)
            x, logpx = @inferred Pathfinder.rand_and_logpdf(rng, dist, ndraws)
            rng = MersenneTwister(42)
            x2 = rand(rng, dist, ndraws)
            logpx2 = logpdf.(dist, x2)
            @test x ≈ x2
            @test logpx ≈ logpx2
        end
    end
end
