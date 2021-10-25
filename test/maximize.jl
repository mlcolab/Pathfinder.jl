using ForwardDiff
using LinearAlgebra
using Optim
using Pathfinder
using Test

include("test_utils.jl")

@testset "maximize_with_trace" begin
    n = 10
    P = inv(rand_pd_mat(Float64, n))
    μ = randn(n)
    f(x) = -dot(x - μ, P, x - μ) / 2
    ∇f(x) = ForwardDiff.gradient(f, x)
    x0 = randn(n)

    @testset "$Topt" for Topt in (Optim.BFGS, Optim.LBFGS, Optim.ConjugateGradient)
        optimizer = Topt()
        xs, fxs, ∇fxs = Pathfinder.maximize_with_trace(f, ∇f, x0, optimizer)
        @test xs[1] ≈ x0
        @test xs[end] ≈ μ
        @test fxs ≈ f.(xs)
        @test ∇fxs ≈ ∇f.(xs)
    end
end
