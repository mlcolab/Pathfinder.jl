# utility functions to work with Transducers.jl

_is_thread_safe_rng(rng) = false
if VERSION ≥ v"1.7.0"
    _is_thread_safe_rng(rng::Random.TaskLocalRNG) = true
    _is_thread_safe_rng(rng::typeof(Random.GLOBAL_RNG)) = true
end

function _default_executor(rng; kwargs...)
    if _is_thread_safe_rng(rng)
        return Transducers.PreferParallel(; kwargs...)
    else
        return Transducers.SequentialEx()
    end
end

# transducer-friendly findmax
function _findmax(x)
    return Transducers.foldxl(x |> Transducers.Enumerate(); init=missing) do xmax_imax, i_xi
        xmax_imax === missing && return reverse(i_xi)
        return last(i_xi) > first(xmax_imax) ? reverse(i_xi) : xmax_imax
    end
end

# WARNING: Type piracy!
# https://github.com/JuliaFolds/Transducers.jl/issues/521
Base.size(x::Transducers.ProgressLoggingFoldable) = size(x.foldable)
