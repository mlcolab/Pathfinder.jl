# Internal helpers

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
