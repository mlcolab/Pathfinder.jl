module PathfinderTuringExt

using Accessors: Accessors
using ADTypes: ADTypes
using DynamicPPL: DynamicPPL
using MCMCChains: MCMCChains
using Pathfinder: Pathfinder
using Random: Random
using Turing: Turing

"""
    create_log_density_problem(model::DynamicPPL.Model, adtype::ADTypes.AbstractADType)

Create a log density problem from a `model`.

The return value is an object implementing the LogDensityProblems API whose log-density is
that of the `model` transformed to unconstrained space with the appropriate log-density
adjustment due to change of variables.
"""
function create_log_density_problem(model::DynamicPPL.Model, adtype::ADTypes.AbstractADType)
    # create an unconstrained VarInfo
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    # DefaultContext ensures that the log-density adjustment is computed
    @static if pkgversion(DynamicPPL) < v"0.35.0"
        prob = DynamicPPL.LogDensityFunction(model, varinfo, DynamicPPL.DefaultContext())
    elseif pkgversion(DynamicPPL) < v"0.37.0"
        prob = DynamicPPL.LogDensityFunction(
            model, varinfo, DynamicPPL.DefaultContext(); adtype
        )
    else
        prob = DynamicPPL.LogDensityFunction(
            model, DynamicPPL.getlogjoint_internal, varinfo; adtype
        )
    end
    return prob
end

function _adtype(prob::DynamicPPL.LogDensityFunction, adtype::ADTypes.AbstractADType)
    hasfield(typeof(prob), :adtype) || return adtype
    return prob.adtype
end

"""
    draws_to_chains(model::DynamicPPL.Model, draws) -> MCMCChains.Chains

Convert a `(nparams, ndraws)` matrix of unconstrained `draws` to a
[`MCMCChains.Chains`](@extref) object with corresponding constrained draws and names
according to `model`.
"""
function draws_to_chains(model::DynamicPPL.Model, draws::AbstractMatrix)
    varinfo = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    draw_con_varinfos = map(eachcol(draws)) do draw
        # this re-evaluates the model but allows supporting dynamic bijectors
        # https://github.com/TuringLang/Turing.jl/issues/2195
        draw_varinfo = DynamicPPL.unflatten(varinfo, draw)
        unlinked_params = DynamicPPL.values_as_in_model(model, true, draw_varinfo)
        iters = map(
            DynamicPPL.varname_and_value_leaves,
            keys(unlinked_params),
            values(unlinked_params),
        )
        return mapreduce(collect, vcat, iters)
    end
    param_con_names = map(first, first(draw_con_varinfos))
    draws_con = reduce(
        vcat, Iterators.map(transpose ∘ Base.Fix1(map, last), draw_con_varinfos)
    )
    chns = MCMCChains.Chains(draws_con, param_con_names)
    return chns
end

@static if isdefined(DynamicPPL, :AbstractInitStrategy)
    struct InitStrategySampler{M<:DynamicPPL.Model,S<:DynamicPPL.AbstractInitStrategy}
        model::M
        strategy::S
    end
    function (spl::InitStrategySampler)(rng::Random.AbstractRNG, point::AbstractVector)
        (; model, strategy) = spl
        varinfo = DynamicPPL.VarInfo(rng, model, strategy)
        varinfo_linked = DynamicPPL.link(varinfo, model)
        copyto!(point, varinfo_linked[:])
    end

    function _maybe_add_sampler_to_kwargs(
        model::DynamicPPL.Model; kwargs...
    )
        # TODO: Change to `InitFromPrior()` (breaking)
        haskey(kwargs, :init_sampler) || return kwargs
        init_sampler = kwargs[:init_sampler]
        if init_sampler isa DynamicPPL.AbstractInitStrategy
            return _merge_keywords(
                kwargs; init_sampler=InitStrategySampler(model, init_sampler)
            )
        else
            return _merge_keywords(kwargs; init_sampler)
        end
    end

    function _format_init(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        init::DynamicPPL.AbstractInitStrategy,
    )
        varinfo = DynamicPPL.VarInfo(rng, model, init)
        varinfo_linked = DynamicPPL.link(varinfo, model)
        return varinfo_linked[:]
    end
    function _format_init(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        init::AbstractVector{<:DynamicPPL.AbstractInitStrategy},
    )
        return map(x -> _format_init(rng, model, x), init)
    end
else
    _maybe_add_sampler_to_kwargs(model::DynamicPPL.Model; kwargs...) = kwargs
end

_format_init(rng::Random.AbstractRNG, model::DynamicPPL.Model, init) = init

function _maybe_add_init_to_kwargs(
    rng::Random.AbstractRNG, model::DynamicPPL.Model; kwargs...
)
    haskey(kwargs, :init) || return kwargs
    return _merge_keywords(kwargs; init=_format_init(rng, model, kwargs[:init]))
end

function _update_kwargs(rng::Random.AbstractRNG, model::DynamicPPL.Model; kwargs...)
    return _maybe_add_init_to_kwargs(
        rng, model; _maybe_add_sampler_to_kwargs(model; kwargs...)...
    )
end

@inline function _merge_keywords(kwargs::Base.Pairs; new_kwargs...)
    return pairs(merge(values(kwargs), values(new_kwargs)))
end

"""
    pathfinder(model::DynamicPPL.Model; kwargs...) -> PathfinderResult

Run single-path Pathfinder on a Turing `model`.

# Arguments
- `model::`[`DynamicPPL.Model`](@extref): Turing/DynamicPPL model whose log-density will be
    maximized.

# Keywords
- `init::`[`InitFromParams`](@extref DynamicPPL.InitFromParams): Initial model parameters.
    If not provided, `init_sampler` is used.
- `init_sampler::`[`AbstractInitStrategy`](@extref DynamicPPL.AbstractInitStrategy): A model
    parameter initialization strategy. If not provided, a uniform sampler over the range
    `[-init_scale, init_scale]` in unconstrained space is used.
- `init_scale::Real=2`: Scale of the default initial point sampler (in unconstrained space).
- Remaining keywords are forwarded to the base method [`pathfinder`](@ref Pathfinder.pathfinder).

# Returns
- [`PathfinderResult`](@ref Pathfinder.PathfinderResult) where `draws_transformed` is an
    [`MCMCChains.Chains`](@extref) with constrained parameter values corresponding to the
    unconstrained `draws`.

# Example

```jldoctest
julia> using Pathfinder, Turing, StableRNGs

julia> rng = StableRNG(42);

julia> @model function demo_model()
           α ~ Normal(0, 1)
           β ~ Beta(5, 1)
           σ ~ truncated(Normal(); lower=0)
       end;

julia> init = InitFromParams((; α=0.0));

julia> init_sampler = InitFromPrior();

julia> result = pathfinder(demo_model(); rng, init, init_sampler)
Single-path Pathfinder result
  tries: 1
  draws: 5
  fit iteration: 6 (total: 6)
  fit ELBO: 0.05 ± 0.09
  fit distribution: MvNormal{Float64, Pathfinder.WoodburyPDMat{Float64, LinearAlgebra.Diagonal{Float64, Vector{Float64}}, Matrix{Float64}, Matrix{Float64}, Pathfinder.WoodburyPDFactorization{Float64, LinearAlgebra.Diagonal{Float64, Vector{Float64}}, LinearAlgebra.QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}, LinearAlgebra.UpperTriangular{Float64, Matrix{Float64}}}}, Vector{Float64}}(
dim: 3
μ: [1.0809306071496012e-24, 1.6094379124340998, 1.2122841002599141e-14]
Σ: [0.7589486918452915 0.0 0.0; 0.0 1.200004838811716 -1.604213692347079e-5; 0.0 -1.6042136923470358e-5 0.5000618385825155]
)


julia> result.draws_transformed
Chains MCMC chain (5×3×1 Array{Float64, 3}):

Iterations        = 1:1:5
Number of chains  = 1
Samples per chain = 5
parameters        = α, β, σ

Use `describe(chains)` for summary statistics and quantiles.

```
"""
Pathfinder.pathfinder(::DynamicPPL.Model; kwargs...)
function Pathfinder.pathfinder(
    model::DynamicPPL.Model;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.pathfinder(
        log_density_problem;
        input=model,
        adtype=new_adtype,
        rng,
        _update_kwargs(rng, model; kwargs...)...,
    )

    # add transformed draws as Chains
    chains_info = (; pathfinder_result=result)
    chains = Accessors.@set draws_to_chains(model, result.draws).info = chains_info
    result_new = Accessors.@set result.draws_transformed = chains
    return result_new
end

"""
    multipathfinder(model::DynamicPPL.Model, ndraws::Int; kwargs...) -> MultiPathfinderResult

Run multi-path Pathfinder on a Turing `model`.

# Arguments
- `model::`[`DynamicPPL.Model`](@extref): Turing/DynamicPPL model whose log-density will be
    maximized.
- `ndraws::Int`: Number of draws to return after (optional) importance resampling.

# Keywords
- `init`: A length `nruns` vector of [`InitFromParams`](@extref DynamicPPL.InitFromParams)
    containing initial model parameters. If not provided, `nruns` is required and
    `init_sampler` is used.
- `nruns::Int`: Number of runs of Pathfinder to perform. Ignored if `init` is provided.
- Remaining keywords are forwarded to the base method
    [`multipathfinder`](@ref Pathfinder.multipathfinder) and
    [`pathfinder(model; kwargs...)`](@ref Pathfinder.pathfinder(::DynamicPPL.Model; kwargs...)).

# Returns
- [`MultiPathfinderResult`](@ref Pathfinder.MultiPathfinderResult) where `draws_transformed`
    (and each single-path result's `draws_transformed`) is an
    [`MCMCChains.Chains`](@extref) with constrained parameter values corresponding to the
    unconstrained `draws`.

# Example

```jldoctest
julia> using Pathfinder, Turing, StableRNGs

julia> rng = StableRNG(42);

julia> @model function demo_model()
           α ~ Normal(0, 1)
           β ~ Beta(5, 1)
           σ ~ truncated(Normal(); lower=0)
       end;

julia> init = [InitFromParams((; α)) for α in -4.0:4.0];

julia> result = multipathfinder(
           demo_model(), 1_000; rng, init, init_sampler=InitFromPrior(),
       )
Multi-path Pathfinder result
  runs: 9
  draws: 1000
  Pareto shape diagnostic: 0.59 (ok)

julia> result.draws_transformed
Chains MCMC chain (1000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
parameters        = α, β, σ

Use `describe(chains)` for summary statistics and quantiles.

```
"""
Pathfinder.multipathfinder(::DynamicPPL.Model, ::Int; kwargs...)
function Pathfinder.multipathfinder(
    model::DynamicPPL.Model,
    ndraws::Int;
    adtype::ADTypes.AbstractADType=Pathfinder.default_ad(),
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...,
)
    log_density_problem = create_log_density_problem(model, adtype)
    new_adtype = _adtype(log_density_problem, adtype)
    result = Pathfinder.multipathfinder(
        log_density_problem,
        ndraws;
        input=model,
        adtype=new_adtype,
        rng,
        _update_kwargs(rng, model; kwargs...)...,
    )

    # add transformed draws as Chains
    chains_info = (; pathfinder_result=result)
    chains = Accessors.@set draws_to_chains(model, result.draws).info = chains_info

    # add transformed draws as Chains for each individual path
    single_path_results_new = map(result.pathfinder_results) do r
        single_chains_info = (; pathfinder_result=r)
        single_chains = Accessors.@set draws_to_chains(model, r.draws).info =
            single_chains_info
        r_new = Accessors.@set r.draws_transformed = single_chains
        return r_new
    end

    result_new = Accessors.@set (Accessors.@set result.draws_transformed = chains).pathfinder_results =
        single_path_results_new
    return result_new
end

end  # module
