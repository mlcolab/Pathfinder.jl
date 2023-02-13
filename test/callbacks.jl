using ForwardDiff, Optim, Pathfinder, Test, UUIDs

include("test_utils.jl")

struct DummyCallback{C,D}
    cond1::C
    cond2::D
end

function (cb::DummyCallback)(x, fx, args...)
    return cb.cond1(x) || cb.cond2(fx)
end

@testset "callbacks" begin
    @testset "CallbackSequence" begin
        cb = Pathfinder.CallbackSequence(
            DummyCallback(iszero, iszero), DummyCallback(isnan, Base.Fix1(any, isnan))
        )
        @inferred cb(1.0, [2.0, 3.0], nothing)
        @test !cb(1.0, [2.0, 3.0], nothing)
        @test cb(0.0, [2.0, 3.0], nothing)
        @test cb(1.0, [0.0, 0.0])
        @test cb(NaN, [1.0, 2.0])
        @test cb(1.0, [NaN, 2.0])
    end

    @testset "CheckFiniteValueCallback" begin
        x = randn(3)
        cb = Pathfinder.CheckFiniteValueCallback(true)
        @inferred cb(x, 1.0, nothing)
        @test !cb(x, 1.0)
        @test !cb(x, 2.0)
        @test cb(x, NaN)
        @test cb(x, -Inf)

        cb2 = Pathfinder.CheckFiniteValueCallback(false)
        @inferred cb2(x, 1.0, nothing)
        @test !cb2(x, 1.0)
        @test !cb2(x, 2.0)
        @test !cb2(x, NaN)
        @test !cb2(x, -Inf)
    end

    @testset "FillTraceCallback" begin
        trace = Pathfinder.OptimizationTrace(
            Vector{Float64}[], Float64[], Vector{Float64}[]
        )
        grad(x) = x / 4
        cb = Pathfinder.FillTraceCallback(grad, trace)
        xs = [randn(10) for _ in 1:3]
        fxs = rand(3)
        for i in eachindex(xs, fxs)
            @test !cb(xs[i], fxs[i], nothing)
            @test length(trace.points) ==
                length(trace.log_densities) ==
                length(trace.gradients) ==
                i
            @test trace.points[i] == xs[i]
            @test trace.log_densities[i] == -fxs[i]
            @test trace.gradients[i] == -grad(xs[i])
        end
    end

    @testset "ProgressCallback" begin
        @testset for maxiters in [10, 100], try_id in 1:2
            progress_id = UUIDs.uuid1()
            progress_trace = []
            reporter = function (progress_id, maxiters, try_id, iter_id)
                push!(progress_trace, (progress_id, maxiters, try_id, iter_id))
                return nothing
            end
            cb = Pathfinder.ProgressCallback(;
                reporter, progress_id, maxiters, try_id, iter_id=0
            )
            @testset for i in 1:3
                @test !cb(randn(10), rand(), nothing)
                @test length(progress_trace) == i
                @test progress_trace[i] == (progress_id, maxiters, try_id, i - 1)
            end
            cb2 = Pathfinder.ProgressCallback(;
                reporter=nothing, progress_id, maxiters, try_id, iter_id=0
            )
            @test !cb2(randn(10), rand(), nothing)
        end
    end

    @testset "report_progress" begin
        @testset for maxiters in (10, 100), try_id in (1, 2)
            logs, = Test.collect_test_logs(; min_level=ProgressLogging.ProgressLevel) do
                ProgressLogging.progress(; name="Optimizing") do progress_id
                    for iter_id in 0:maxiters
                        Pathfinder.report_progress(progress_id, maxiters, try_id, iter_id)
                    end
                end
            end
            @test length(logs) == maxiters + 3
            @test logs[1].kwargs[:progress] === nothing
            @test logs[1].message == "Optimizing"
            for i in 0:maxiters
                @test logs[i + 2].kwargs[:progress] == i / maxiters
                @test logs[i + 2].message == "Optimizing (try $try_id)"
            end
            @test logs[maxiters + 3].kwargs[:progress] == "done"
        end
    end

    @testset "OptimJLCallback" begin
        f(x) = -logp_banana(x)
        grad(x) = ForwardDiff.gradient(f, x)
        trace = Pathfinder.OptimizationTrace(
            Vector{Float64}[], Float64[], Vector{Float64}[]
        )
        callback = Pathfinder.OptimJLCallbackAdaptor(
            Pathfinder.FillTraceCallback(grad, trace)
        )
        x0 = randn(2)
        options = Optim.Options(; callback, store_trace=true, extended_trace=true)
        sol = Optim.optimize(f, x0, LBFGS(), options)
        @test trace.points == Optim.x_trace(sol)
        @test trace.log_densities == -Optim.f_trace(sol)
        @test trace.gradients ≈ -[t.metadata["g(x)"] for t in sol.trace]

        cb2 = Pathfinder.OptimJLCallbackAdaptor(nothing)
        @test !cb2([])
    end
end
