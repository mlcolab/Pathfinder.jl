function pathfinder(
    model::DynamicPPL.Model;
    rng=Random.GLOBAL_RNG,
    sample_init_scale=2,
    sample_init_fun=UniformSampler(sample_init_scale),
    kwargs...,
)
    prob = Turing.optim_problem(model, Turing.MAP(); constrained=true, init_theta=init)
    sample_init_fun(rng, prob.prob.u0)
    return pathfinder(prob.prob; rng, kwargs...)
end

function multipathfinder(
    model::DynamicPPL.Model, ndraws::Int;
    rng=Random.GLOBAL_RNG,
    sample_init_scale=2,
    sample_init_fun=UniformSampler(sample_init_scale),
    nruns::Int,
    kwargs...,
)
    prob = Turing.optim_function(model, Turing.MAP(); constrained=true)
    init1 = f.init()
    init = [sample_init_fun(rng, init1)]
    for _ in 2:nruns
        push!(init, sample_init_fun(rng, deepcopy(init1)))
    end
    return multipathfinder(prob.func, ndraws; rng, init, kwargs...)
end
