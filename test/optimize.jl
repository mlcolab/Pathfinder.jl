using AbstractDifferentiation
using ForwardDiff
using GalacticOptim
using LinearAlgebra
using NLopt
using Optim
using Pathfinder
using Test
using Transducers

include("test_utils.jl")

@testset "build_optim_function" begin
    n = 20
    f(x) = logp_banana(x)
    ∇f(x) = ForwardDiff.gradient(f, x)
    ad_backend = AD.ForwardDiffBackend()
    x = randn(n)
    funs = [
        "user-provided gradient" => Pathfinder.build_optim_function(f, ∇f; ad_backend),
        "automatic gradient" => Pathfinder.build_optim_function(f; ad_backend),
    ]
    @testset "$name" for (name, fun) in funs
        @test fun isa GalacticOptim.OptimizationFunction
        @test fun.f(x) ≈ -f(x)
        ∇fx = similar(x)
        fun.grad(∇fx, x, nothing)
        @test ∇fx ≈ -∇f(x)
        H = similar(x, n, n)
        fun.hess(H, x, nothing)
        @test H ≈ -ForwardDiff.hessian(f, x)
        Hv = similar(x)
        v = randn(n)
        fun.hv(Hv, x, v, nothing)
        @test Hv ≈ H * v
    end
end

@testset "build_optim_problem" begin
    n = 20
    f(x) = logp_banana(x)
    ∇f(x) = ForwardDiff.gradient(f, x)
    ad_backend = AD.ForwardDiffBackend()
    x0 = randn(n)
    fun = Pathfinder.build_optim_function(f; ad_backend)
    prob = Pathfinder.build_optim_problem(fun, x0)
    @test prob isa GalacticOptim.OptimizationProblem
    @test prob.f === fun
    @test prob.u0 == x0
    @test prob.p === nothing
end

@testset "optimize_with_trace" begin
    n = 10
    P = inv(rand_pd_mat(Float64, n))
    μ = randn(n)
    f(x) = -dot(x - μ, P, x - μ) / 2
    ∇f(x) = ForwardDiff.gradient(f, x)

    x0 = randn(n)
    ad_backend = AD.ForwardDiffBackend()
    fun = Pathfinder.build_optim_function(f; ad_backend)
    prob = Pathfinder.build_optim_problem(fun, x0)

    optimizers = [
        Optim.BFGS(), Optim.LBFGS(), Optim.ConjugateGradient(), NLopt.Opt(:LD_LBFGS, n)
    ]
    @testset "$(typeof(optimizer))" for optimizer in optimizers
        xs, fxs, ∇fxs = Pathfinder.optimize_with_trace(prob, optimizer, SequentialEx())
        @test xs[1] ≈ x0
        @test xs[end] ≈ μ
        @test fxs ≈ f.(xs)
        @test ∇fxs ≈ ∇f.(xs) atol = 1e-4

        if !(optimizer isa NLopt.Opt)
            options = Optim.Options(; store_trace=true, extended_trace=true)
            res = Optim.optimize(
                x -> -f(x), (y, x) -> copyto!(y, -∇f(x)), x0, optimizer, options
            )
            @test Optim.iterations(res) == length(xs) - 1
            @test Optim.x_trace(res) ≈ xs
            @test Optim.minimizer(res) ≈ xs[end]
        end

        xs2, fxs2, ∇fxs2 = Pathfinder.optimize_with_trace(prob, optimizer, ThreadedEx())
        @test xs2 ≈ xs
        @test fxs2 ≈ fxs
        @test ∇fxs2 ≈ ∇fxs
    end
end
