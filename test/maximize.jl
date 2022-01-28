using ForwardDiff
using GalacticOptim
using LinearAlgebra
using Optim
using Pathfinder
using Test

include("test_utils.jl")

@testset "optimization" begin
    n = 10
    P = inv(rand_pd_mat(Float64, n))
    μ = randn(n)
    f(x) = -dot(x - μ, P, x - μ) / 2
    ∇f(x) = ForwardDiff.gradient(f, x)
    x0 = randn(n)
    fun = Pathfinder.build_optim_function(f, ∇f)
    @test fun isa GalacticOptim.OptimizationFunction
    @test fun.grad !== nothing
    prob = Pathfinder.build_optim_problem(fun, x0)
    @test prob isa GalacticOptim.OptimizationProblem
    @test prob.f === fun
    @test prob.u0 == x0

    @testset "$Topt" for Topt in (Optim.BFGS, Optim.LBFGS, Optim.ConjugateGradient)
        optimizer = Topt()
        xs, fxs, ∇fxs = Pathfinder.optimize_with_trace(prob, optimizer)
        options = Optim.Options(; store_trace=true, extended_trace=true)
        res = Optim.optimize(
            x -> -f(x), (y, x) -> copyto!(y, -∇f(x)), x0, optimizer, options
        )
        @test Optim.iterations(res) == length(xs) - 1
        @test Optim.x_trace(res) ≈ xs
        @test Optim.minimizer(res) ≈ xs[end]
        @test xs[1] ≈ x0
        @test xs[end] ≈ μ
        @test fxs ≈ f.(xs)
        @test ∇fxs ≈ ∇f.(xs)
    end
end
