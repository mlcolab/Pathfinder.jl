using LogDensityProblems
using LogDensityProblemsAD
using ForwardDiff
using Pathfinder
using ReverseDiff
using Test

struct LogDensityFunctionWithGrad{F,G}
    logp::F
    ∇logp::G
    dim::Int
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunctionWithGrad})
    return LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.dimension(ℓ::LogDensityFunctionWithGrad) = ℓ.dim
LogDensityProblems.logdensity(ℓ::LogDensityFunctionWithGrad, x) = ℓ.logp(x)
function LogDensityProblems.logdensity_and_gradient(ℓ::LogDensityFunctionWithGrad, x)
    return ℓ.logp(x), ℓ.∇logp(x)
end

@testset "LogDensityProblem creation" begin
    @testset "LogDensityFunction" begin
        f(x) = -sum(abs2, x) / 2

        @testset for dim in (5, 10)
            ℓ = @inferred Pathfinder.LogDensityFunction(f, dim)
            @test ℓ isa Pathfinder.LogDensityFunction{typeof(f)}
            @test LogDensityProblems.capabilities(ℓ) ===
                LogDensityProblems.LogDensityOrder{0}()
            @test LogDensityProblems.dimension(ℓ) == dim
            x = randn(dim)
            @test LogDensityProblems.logdensity(ℓ, x) == f(x)
            LogDensityProblems.stresstest(LogDensityProblems.logdensity, ℓ)
        end
    end

    @testset "_logdensityproblem" begin
        @testset for ad_backend in (:ForwardDiff, :ReverseDiff, Val(:ReverseDiff)),
            n in (5, 10)

            f(x) = -sum(abs2, x) / 2
            ∇f(x) = -x

            @testset for input in (
                f, Pathfinder.LogDensityFunction(f, n), LogDensityFunctionWithGrad(f, ∇f, n)
            )
                dim = (input === f) ? n : -1
                if ad_backend isa Val || input isa LogDensityFunctionWithGrad
                    @inferred Pathfinder._logdensityproblem(input, dim, ad_backend)
                end
                ℓ = Pathfinder._logdensityproblem(input, dim, ad_backend)
                input isa LogDensityFunctionWithGrad && @test ℓ === input
                @test LogDensityProblems.capabilities(ℓ) ===
                    LogDensityProblems.LogDensityOrder{1}()
                @test LogDensityProblems.dimension(ℓ) == n
                x = randn(n)
                @test LogDensityProblems.logdensity(ℓ, x) == f(x)
                LogDensityProblems.stresstest(LogDensityProblems.logdensity, ℓ)
                LogDensityProblems.stresstest(LogDensityProblems.logdensity_and_gradient, ℓ)
                ℓ isa LogDensityFunctionWithGrad && continue
                LogDensityProblems.stresstest(
                    LogDensityProblems.logdensity_gradient_and_hessian, ℓ
                )
            end
        end
    end
end
