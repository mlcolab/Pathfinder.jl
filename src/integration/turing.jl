using .Turing: Turing, DynamicPPL

function pathfinder(
    model::DynamicPPL.Model;
    rng=Random.GLOBAL_RNG,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    init=nothing,
    kwargs...,
)
    prob = Turing.optim_problem(model, Turing.MAP(); constrained=false, init_theta=init)
    init_sampler(rng, prob.prob.u0)
    return pathfinder(prob.prob; rng, kwargs...)
end

function multipathfinder(
    model::DynamicPPL.Model, ndraws::Int;
    rng=Random.GLOBAL_RNG,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    nruns::Int,
    kwargs...,
)
    fun = Turing.optim_function(model, Turing.MAP(); constrained=false)
    init1 = fun.init()
    init = [init_sampler(rng, init1)]
    for _ in 2:nruns
        push!(init, init_sampler(rng, deepcopy(init1)))
    end
    return multipathfinder(fun.func, ndraws; rng, init, kwargs...)
end
