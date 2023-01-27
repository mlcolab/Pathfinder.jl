struct LogDensityFunction{L}
    logp::L
    dim::Int
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end
LogDensityProblems.dimension(ℓ::LogDensityFunction) = ℓ.dim
LogDensityProblems.logdensity(ℓ::LogDensityFunction, x) = ℓ.logp(x)

function _logdensityproblem(logp_or_ℓ, dim, ad_backend)
    ℓ = if LogDensityProblems.capabilities(logp_or_ℓ) === nothing
        LogDensityFunction(logp_or_ℓ, dim)
    else
        logp_or_ℓ
    end
    if LogDensityProblems.capabilities(ℓ) === LogDensityProblems.LogDensityOrder{0}()
        return LogDensityProblemsAD.ADgradient(ad_backend, ℓ)
    else
        return ℓ
    end
end
