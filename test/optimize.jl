using LinearAlgebra
using Optim
using OptimizationNLopt
using Pathfinder
using ProgressLogging
using SciMLBase
using Test

@testset "build_optim_function" begin
    n = 20
    ℓ = build_logdensityproblem(logp_banana, n)
    x = randn(n)
    fun = Pathfinder.build_optim_function(ℓ)
    @test fun isa SciMLBase.OptimizationFunction
    @test SciMLBase.isinplace(fun)
    @test fun.f(x, nothing) ≈ -ℓ.logp(x)
    ∇fx = similar(x)
    fun.grad(∇fx, x, nothing)
    @test ∇fx ≈ -ℓ.∇logp(x)
    H = similar(x, n, n)
    fun.hess(H, x, nothing)
    @test H ≈ -ℓ.∇²logp(x)
end

@testset "build_optim_problem" begin
    n = 20
    ℓ = build_logdensityproblem(logp_banana, n)
    x0 = randn(n)
    fun = Pathfinder.build_optim_function(ℓ)
    prob = Pathfinder.build_optim_problem(fun, x0)
    @test SciMLBase.isinplace(prob)
    @test prob isa SciMLBase.OptimizationProblem
    @test prob.f === fun
    @test prob.u0 == x0
    @test prob.p === nothing
end

@testset "OptimizationCallback" begin
    @testset "callback return value" begin
        progress_name = "Optimizing"
        progress_id = nothing
        maxiters = 1_000
        x = randn(5)
        check_vals = [0.0, NaN, -Inf, Inf]
        @testset for fail_on_nonfinite in [true, false],
            fval in check_vals,
            gval in check_vals,
            cbfail in [true, false]

            xs = Vector{Float64}[]
            fxs = Float64[]
            ∇fxs = Vector{Float64}[]
            ∇f = function (x)
                g = -x
                g[end] = gval
                return g
            end
            should_fail =
                cbfail ||
                (fail_on_nonfinite && (isnan(fval) || fval == Inf || !isfinite(gval)))
            if isdefined(Optimization, :OptimizationState)
                # Optimization v3.21.0 and later
                callback = (state, args...) -> cbfail
                state = Optimization.OptimizationState(;
                    iter=0, u=x, objective=-fval, grad=-∇f(x)
                )
                cb_args = (state, -fval)
            else
                # Optimization v3.20.X and earlier
                callback = (x, fx, args...) -> cbfail
                cb_args = (x, -fval)
            end
            cb = Pathfinder.OptimizationCallback(
                xs,
                fxs,
                ∇fxs,
                ∇f,
                progress_name,
                progress_id,
                maxiters,
                callback,
                fail_on_nonfinite,
            )
            @test cb isa Pathfinder.OptimizationCallback
            @test cb(cb_args...) == should_fail
        end
    end
end

@testset "optimize_with_trace" begin
    n = 10
    P = inv(rand_pd_mat(Float64, n))
    μ = randn(n)
    f(x) = -dot(x - μ, P, x - μ) / 2
    ℓ = build_logdensityproblem(f, n)

    x0 = randn(n)
    fun = Pathfinder.build_optim_function(ℓ)
    prob = Pathfinder.build_optim_problem(fun, x0)

    optimizers = [
        "Optim.BFGS" => Optim.BFGS(),
        "Optim.LBFGS" => Optim.LBFGS(),
        "Optim.ConjugateGradient" => Optim.ConjugateGradient(),
        "NLopt.Opt" => NLopt.Opt(:LD_LBFGS, n),
    ]
    @testset "$name" for (name, optimizer) in optimizers
        optim_sol, optim_trace = Pathfinder.optimize_with_trace(prob, optimizer)
        @test optim_sol isa SciMLBase.OptimizationSolution
        @test optim_trace isa Pathfinder.OptimizationTrace
        @test optim_trace.points[1] ≈ x0
        @test optim_trace.points[end] ≈ μ
        @test optim_sol.u ≈ μ
        @test optim_trace.log_densities ≈ f.(optim_trace.points)
        @test optim_trace.gradients ≈ ℓ.∇logp.(optim_trace.points) atol = 1e-4

        if !(optimizer isa NLopt.Opt)
            options = Optim.Options(; store_trace=true, extended_trace=true)
            res = Optim.optimize(
                x -> -f(x), (y, x) -> copyto!(y, -ℓ.∇logp(x)), x0, optimizer, options
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
