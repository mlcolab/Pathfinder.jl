var documenterSearchIndex = {"docs":
[{"location":"multipathfinder/#Multi-path-Pathfinder","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"","category":"section"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"multipathfinder","category":"page"},{"location":"multipathfinder/#Pathfinder.multipathfinder","page":"Multi-path Pathfinder","title":"Pathfinder.multipathfinder","text":"multipathfinder(\n    logp,\n    [∇logp,]\n    θ₀s::AbstractVector{AbstractVector{<:Real}},\n    ndraws::Int;\n    kwargs...\n)\n\nFilter samples from a mixture of multivariate normal distributions fit using pathfinder.\n\nFor nruns=length(θ₀s), nruns parallel runs of pathfinder produce nruns multivariate normal approximations q_k = q(phi  mu_k Sigma_k) of the posterior. These are combined to a mixture model q with uniform weights.\n\nq is augmented with the component index to generate random samples, that is, elements (k phi) are drawn from the augmented mixture model\n\ntildeq(phi k  mu Sigma) = K^-1 q(phi  mu_k Sigma_k)\n\nwhere k is a component index, and K= nruns. These draws are then resampled with replacement. Discarding k from the samples would reproduce draws from q.\n\nIf importance=true, then Pareto smoothed importance resampling is used, so that the resulting draws better approximate draws from the target distribution p instead of q.\n\nArguments\n\nlogp: a callable that computes the log-density of the target distribution.\n∇logp: a callable that computes the gradient of logp. If not provided, logp is   automatically differentiated using the backend specified in ad_backend.\nθ₀s::AbstractVector{AbstractVector{<:Real}}: vector of length nruns of initial points   of length dim from which each single-path Pathfinder run will begin\nndraws::Int: number of approximate draws to return\n\nKeywords\n\nad_backend=AD.ForwardDiffBackend(): AbstractDifferentiation.jl AD backend.\nndraws_per_run::Int=5: The number of draws to take for each component before resampling.\nimportance::Bool=true: Perform Pareto smoothed importance resampling of draws.\nrng::AbstractRNG=Random.GLOBAL_RNG: Pseudorandom number generator. It is recommended to   use a parallelization-friendly PRNG like the default PRNG on Julia 1.7 and up.\nexecutor::Transducers.Executor: Transducers.jl executor that determines if and how   to run the single-path runs in parallel. If rng is known to be thread-safe, the   default is Transducers.PreferParallel(; basesize=1) (parallel executation, defaulting   to multi-threading). Otherwise, it is Transducers.SequentialEx() (no parallelization).\nexecutor_per_run::Transducers.Executor=Transducers.SequentialEx(): Transducers.jl   executor used within each run to parallelize PRNG calls. Defaults to no parallelization.   See pathfinder for a description.\nkwargs... : Remaining keywords are forwarded to pathfinder.\n\nReturns\n\nq::Distributions.MixtureModel: Uniformly weighted mixture of ELBO-maximizing   multivariate normal distributions\nϕ::AbstractMatrix{<:Real}: approximate draws from target distribution with size   (dim, ndraws)\ncomponent_inds::Vector{Int}: Indices k of components in q from which each column   in ϕ was drawn.\n\n\n\n\n\nmultipathfinder(\n    f::GalacticOptim.OptimizationFunction,\n    θ₀s::AbstractVector{<:Real},\n    ndraws::Int;\n    kwargs...,\n)\n\nFilter samples from a mixture of multivariate normal distributions fit using pathfinder.\n\nf is a user-created optimization function that represents the negative log density with its gradient and must have the necessary features (e.g. a Hessian function or specified automatic differentiation type) for the chosen optimization algorithm. For details, see GalacticOptim.jl: OptimizationFunction.\n\nSee multipathfinder for a description of remaining arguments.\n\n\n\n\n\n","category":"function"},{"location":"multipathfinder/#Examples","page":"Multi-path Pathfinder","title":"Examples","text":"","category":"section"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"Especially for complicated target distributions, it's more useful to run multi-path Pathfinder. One difficult distribution to sample is Neal's funnel:","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"tau sim mathrmNormal(0 3)\nbeta_i sim mathrmNormal(0 e^tau2)","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"Such funnel geometries appear in other models and typically frustrate MCMC sampling. Multi-path Pathfinder can't sample the funnel well, but it can quickly give us draws that can help us diagnose that we have a funnel.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"In this example, we draw from a 100-dimensional funnel and visualize 2 dimensions.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"using ForwardDiff, Pathfinder, Random\n\nRandom.seed!(42)\n\nfunction logp(x)\n    n = length(x)\n    τ = x[1]\n    β = view(x, 2:n)\n    return ((τ / 3)^2 + (n - 1) * τ + sum(b -> abs2(b * exp(-τ / 2)), β)) / -2\nend\n∇logp(x) = ForwardDiff.gradient(logp, x)\n\ndim = 100\nndraws = 1_000\nnruns = 20\nnothing # hide","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"Next, we create initial points for the Pathfinder runs.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"θ₀s = collect.(eachcol(rand(dim, nruns) .* 20 .- 10));","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"Now we run multi-path Pathfinder.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"ndraws_per_run = ndraws ÷ nruns\n@time q, ϕ, component_ids = multipathfinder(logp, ∇logp, θ₀s, ndraws; ndraws_per_run);","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"The first return value is a uniformly-weighted Distributions.MixtureModel. Each component is the result of a single Pathfinder run.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"typeof(q)","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"While we could draw samples from q directly, these aren't equivalent to the samples returned by multi-path Pathfinder, which uses multiple importance sampling with Pareto-smoothed importance resampling to combine the individual runs.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"Note that PSIS.jl, which smooths the importance weights, warns us that the importance weights are unsuitable for computing estimates, so we should definitely run MCMC to get better samples. Pathfinder's samples can still be useful though for initializing MCMC.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"Let's compare Pathfinder's samples with samples from q directly, plotted over the exact marginal density.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"using Plots\n\nτ_approx = ϕ[1, :]\nβ₁_approx = ϕ[2, :]\nϕ2 = rand(q, ndraws)\nτ_approx2 = ϕ2[1, :]\nβ₁_approx2 = ϕ2[2, :]\n\nτ_range = -15:0.01:5\nβ₁_range = -5:0.01:5\n\nplt = contour(β₁_range, τ_range, (β, τ) -> exp(logp([τ, β])))\nscatter!(β₁_approx2, τ_approx2; msw=0, ms=2, alpha=0.3, color=:blue, label=\"Pathfinder w/o IR\")\nscatter!(β₁_approx, τ_approx; msw=0, ms=2, alpha=0.5, color=:red, label=\"Pathfinder\")\nplot!(xlims=extrema(β₁_range), ylims=extrema(τ_range), xlabel=\"β₁\", ylabel=\"τ\", legend=true)\nplt","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"As expected from PSIS.jl's warning, neither set of samples cover the density well. But we can also see that the mixture model (q) places too many samples in regions of low density, which is corrected by the importance resampling.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"We can check how much each component contributed to the returned sample.","category":"page"},{"location":"multipathfinder/","page":"Multi-path Pathfinder","title":"Multi-path Pathfinder","text":"histogram(component_ids; bins=(0:nruns) .+ 0.5, bar_width=0.8, xticks=1:nruns,\n          xlabel=\"Component index\", ylabel=\"Count\", legend=false)","category":"page"},{"location":"pathfinder/#Single-path-Pathfinder","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"","category":"section"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"pathfinder","category":"page"},{"location":"pathfinder/#Pathfinder.pathfinder","page":"Single-path Pathfinder","title":"Pathfinder.pathfinder","text":"pathfinder(logp[, ∇logp], θ₀::AbstractVector{<:Real}, ndraws::Int; kwargs...)\n\nFind the best multivariate normal approximation encountered while maximizing logp.\n\nFrom an optimization trajectory, Pathfinder constructs a sequence of (multivariate normal) approximations to the distribution specified by logp. The approximation that maximizes the evidence lower bound (ELBO), or equivalently, minimizes the KL divergence between the approximation and the true distribution, is returned.\n\nThe covariance of the multivariate normal distribution is an inverse Hessian approximation constructed using at most the previous history_length steps.\n\nArguments\n\nlogp: a callable that computes the log-density of the target distribution.\n∇logp: a callable that computes the gradient of logp. If not provided, logp is   automatically differentiated using the backend specified in ad_backend.\nθ₀: initial point of length dim from which to begin optimization\nndraws: number of approximate draws to return\n\nKeywords\n\nad_backend=AD.ForwardDiffBackend(): AbstractDifferentiation.jl AD backend.\nrng::Random.AbstractRNG: The random number generator to be used for drawing samples\nexecutor::Transducers.Executor=Transducers.SequentialEx(): Transducers.jl executor that   determines if and how to perform ELBO computation in parallel. The default   (SequentialEx()) performs no parallelization. If rng is known to be thread-safe, and   the log-density function is known to have no internal state, then   Transducers.PreferParallel() may be used to parallelize log-density evaluation.   This is generally only faster for expensive log density functions.\noptimizer: Optimizer to be used for constructing trajectory. Can be any optimizer   compatible with GalacticOptim, so long as it supports callbacks. Defaults to   Optim.LBFGS(; m=6, linesearch=LineSearches.MoreThuente()). See   the GalacticOptim.jl documentation for details.\nhistory_length::Int=6: Size of the history used to approximate the   inverse Hessian. This should only be set when optimizer is not an Optim.LBFGS.\nndraws_elbo::Int=5: Number of draws used to estimate the ELBO\nkwargs... : Remaining keywords are forwarded to GalacticOptim.OptimizationProblem.\n\nReturns\n\nq::Distributions.MvNormal: ELBO-maximizing multivariate normal distribution\nϕ::AbstractMatrix{<:Real}: draws from multivariate normal with size (dim, ndraws)\nlogqϕ::Vector{<:Real}: log-density of multivariate normal at columns of ϕ\n\n\n\n\n\npathfinder(\n    f::GalacticOptim.OptimizationFunction,\n    θ₀::AbstractVector{<:Real},\n    ndraws::Int;\n    kwargs...,\n)\n\nFind the best multivariate normal approximation encountered while minimizing f.\n\nf is a user-created optimization function that represents the negative log density with its gradient and must have the necessary features (e.g. a Hessian function or specified automatic differentiation type) for the chosen optimization algorithm. For details, see GalacticOptim.jl: OptimizationFunction.\n\nSee pathfinder for a description of remaining arguments.\n\n\n\n\n\npathfinder(prob::GalacticOptim.OptimizationProblem, ndraws::Int; kwargs...)\n\nFind the best multivariate normal approximation encountered while solving prob.\n\nprob is a user-created optimization problem that represents the negative log density with its gradient, an initial position and must have the necessary features (e.g. a Hessian function or specified automatic differentiation type) for the chosen optimization algorithm. For details, see GalacticOptim.jl: Defining OptimizationProblems.\n\nSee pathfinder for a description of remaining arguments.\n\n\n\n\n\n","category":"function"},{"location":"pathfinder/#Examples","page":"Single-path Pathfinder","title":"Examples","text":"","category":"section"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"For a simple example, we'll run Pathfinder on a multivariate normal distribution with a dense covariance matrix.","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"using LinearAlgebra, Pathfinder, Random\nRandom.seed!(42)\n\nΣ = [\n    2.71   0.5    0.19   0.07   1.04\n    0.5    1.11  -0.08  -0.17  -0.08\n    0.19  -0.08   0.26   0.07  -0.7\n    0.07  -0.17   0.07   0.11  -0.21\n    1.04  -0.08  -0.7   -0.21   8.65\n]\nμ = [-0.55, 0.49, -0.76, 0.25, 0.94]\nP = inv(Symmetric(Σ))\n\nfunction logp(x)\n    z = x - μ\n    return -dot(z, P, z) / 2;\nend\n∇logp(x) = P * (μ - x)\nnothing # hide","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"Next, we create initial points for the Pathfinder runs.","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"θ₀ = rand(5) .* 4 .- 2  # θ₀ ~ Uniform(-2, 2);","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"Now we run Pathfinder.","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"@time q, ϕ, logqϕ = pathfinder(logp, ∇logp, θ₀, 100; ndraws_elbo=100);","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"The first return value is a multivariate normal approximation to our target distribution. Its mean and covariance are quite close to our target distribution's.","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"q\nq.μ\nq.Σ","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"ϕ is a Matrix whose columns are the requested draws from q:","category":"page"},{"location":"pathfinder/","page":"Single-path Pathfinder","title":"Single-path Pathfinder","text":"ϕ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Pathfinder","category":"page"},{"location":"#Pathfinder","page":"Home","title":"Pathfinder","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pathfinder[Zhang2021] is a variational method for initializing Markov chain Monte Carlo (MCMC) methods.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"When using MCMC to draw samples from some target distribution, there is often a length warm-up phase with 2 goals:","category":"page"},{"location":"","page":"Home","title":"Home","text":"converge to the typical set (the region of the target distribution where the bulk of the probability mass is located)\nadapt any tunable parameters of the MCMC sampler (optional)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Typically (2) requires a lengthy exploration of the typical set. An example is the widely used windowed adaptation scheme of Hamiltonian Monte Carlo (HMC), where a step size and mass matrix are adapted For posteriors with complex geometry, the adaptation phase can require many evaluations of the gradient of the log density function of the target distribution.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pathfinder attempts to return samples in or near the typical set with many fewer gradient evaluations. Pathfinder uses L-BFGS to construct a maximum a posteriori (MAP) estimate of a target distribution p. It then uses the trace of the optimization to construct a sequence of multivariate normal approximations to the target distribution, returning the approximation that maximizes the evidence lower bound (ELBO), as well as draws from the distribution. The covariance of the multivariate normal approximation can be used to instantiate the mass matrix adaptation in HMC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Its extension, multi-path Pathfinder, runs Pathfinder multiple times. It returns a uniformly-weighted mixture model of the multivariate normal approximations of the individual runs. It also uses importance resampling to return samples that better approximate the target distribution.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[Zhang2021]: Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2021).           Pathfinder: Parallel quasi-Newton variational inference.           arXiv: 2108.03782 [stat.ML].           Code","category":"page"}]
}
