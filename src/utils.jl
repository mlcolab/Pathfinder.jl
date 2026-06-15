# Internal helpers

# Number of parallel chunks for `len` items given `ntasks`; `1` runs sequentially.
function _nchunks(len::Int, ntasks::Int)
    nchunks = min(len, ntasks)
    if nchunks > 1 && Threads.nthreads() > 1
        return nchunks
    else
        return 1
    end
end

# Map `f(s, xs[i]...)` over the elements of the equal-axes arrays `xss`, using at most
# `ntasks` parallel tasks and per-chunk state `s = setup()` built once per chunk.
function _chunk_tmap(f, xss::AbstractArray...; ntasks::Int, setup)
    nchunks = _nchunks(length(first(xss)), ntasks)
    if nchunks == 1
        let s = setup()
            return map((xs...) -> f(s, xs...), xss...)
        end
    else
        return OhMyThreads.tmapreduce(
            vcat, OhMyThreads.chunks(eachindex(xss...); n=nchunks); chunking=false
        ) do chunk
            let s = setup()
                return map(i -> f(s, map(xs -> xs[i], xss)...), chunk)
            end
        end
    end
end

# `map`/`mapreduce` using at most `ntasks` parallel tasks.
function _maybe_tmap(f, xs::AbstractArray, ntasks::Int)
    nchunks = _nchunks(length(xs), ntasks)
    if nchunks == 1
        return map(f, xs)
    else
        return OhMyThreads.tmap(f, xs; nchunks)
    end
end

function _maybe_tmapreduce(f, op, xs::AbstractArray, ntasks::Int)
    nchunks = _nchunks(length(xs), ntasks)
    if nchunks == 1
        return mapreduce(f, op, xs)
    else
        return OhMyThreads.tmapreduce(f, op, xs; nchunks)
    end
end

# Return `(maxvalue, argmaxindex)` of `key.(xs)`, skipping `NaN`s. If the
# first observed value is `NaN`, it is retained until a non-`NaN` entry
# replaces it; if every entry is `NaN`, the first one is returned (matching
# the semantics the success check in `singlepath.jl` relies on).
_findmax_skipnan(xs) = _findmax_skipnan(identity, xs)

function _findmax_skipnan(key, xs)
    state = missing
    for (i, x) in enumerate(xs)
        xi = key(x)
        if state === missing
            state = (xi, i)
            continue
        end
        isnan(xi) && continue
        xmax = first(state)
        if isnan(xmax) || xi > xmax
            state = (xi, i)
        end
    end
    return state
end
