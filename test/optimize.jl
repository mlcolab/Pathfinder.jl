using AbstractDifferentiation
using ForwardDiff
using LinearAlgebra
using Optim
using OptimizationNLopt
using Pathfinder
using ProgressLogging
using SciMLBase
using Test

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
        @test fun isa SciMLBase.OptimizationFunction
        @test SciMLBase.isinplace(fun)
        @test fun.f(x, nothing) ≈ -f(x)
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
    @test prob isa SciMLBase.OptimizationProblem
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
        optim_sol, optim_trace = Pathfinder.optimize_with_trace(prob, optimizer)
        @test optim_sol isa SciMLBase.OptimizationSolution
        @test optim_trace isa Pathfinder.OptimizationTrace
        @test optim_trace.points[1] ≈ x0
        @test optim_sol.prob.u0 ≈ x0
        @test optim_trace.points[end] ≈ μ
        @test optim_sol.u ≈ μ
        @test optim_trace.log_densities ≈ f.(optim_trace.points)
        @test optim_trace.gradients ≈ ∇f.(optim_trace.points) atol = 1e-4

        if !(optimizer isa NLopt.Opt)
            options = Optim.Options(; store_trace=true, extended_trace=true)
            res = Optim.optimize(
                x -> -f(x), (y, x) -> copyto!(y, -∇f(x)), x0, optimizer, options
            )
            @test Optim.iterations(res) == length(optim_trace.points) - 1
            @test Optim.x_trace(res) ≈ optim_trace.points
            @test Optim.minimizer(res) ≈ optim_trace.points[end]
        end
    end

    @testset "progress logging" begin
        logs, = Test.collect_test_logs(; min_level=ProgressLogging.ProgressLevel) do
            ProgressLogging.progress(; name="Optimizing") do progress_id
                Pathfinder.optimize_with_trace(prob, Optim.LBFGS(); progress_id)
            end
        end
        @test logs[1].kwargs[:progress] === nothing
        @test logs[1].message == "Optimizing"
        @test logs[2].kwargs[:progress] == 0.0
        @test logs[2].message == "Optimizing"
        @test logs[3].kwargs[:progress] == 0.001
        @test logs[end].kwargs[:progress] == "done"
    end
end
