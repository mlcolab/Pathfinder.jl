function maximize_with_trace(f, ∇f, x₀, optimizer; kwargs...)
    negf(x) = -f(x)
    g!(y, x) = (y .= .-∇f(x))

    function callback(states)
        # terminate BFGS if optimization encounters NaNs
        s = states[end]
        md = s.metadata
        return isnan(s.value) || any(isnan, md["x"]) || any(isnan, md["g(x)"])
    end
    options = Optim.Options(;
        store_trace=true, extended_trace=true, callback=callback, kwargs...
    )
    res = Optim.optimize(negf, g!, x₀, optimizer, options)

    xs = Optim.x_trace(res)::Vector{typeof(Optim.minimizer(res))}
    fxs = -Optim.f_trace(res)
    ∇fxs = map(tr -> -tr.metadata["g(x)"], Optim.trace(res))::typeof(xs)

    return xs, fxs, ∇fxs
end
