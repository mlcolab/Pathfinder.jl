# Using Pathfinder to initializing sampling with Turing

If you have a _differentiable_ Turing model, then you can use Pathfinder to initialize its parameters.
To demonstrate this, we'll use the linear regression tutorial from the Turing docs.

First, we load the packages.

```@example 1
using Pathfinder, Random, RDatasets, Turing
using MLDataUtils: shuffleobs, splitobs, rescale!
Turing.setprogress!(false);
nothing # hide
```

Then we load and standardize the data.

```@example 1
data = RDatasets.dataset("datasets", "mtcars");

select!(data, Not(:Model))

# Split our dataset 70%/30% into training/test sets.
trainset, testset = splitobs(shuffleobs(data), 0.7)

# Turing requires data in matrix form.
target = :MPG
train = Matrix(select(trainset, Not(target)))
test = Matrix(select(testset, Not(target)))
train_target = trainset[:, target]
test_target = testset[:, target]

# Standardize the features.
μ, σ = rescale!(train; obsdim = 1)
rescale!(test, μ, σ; obsdim = 1)

# Standardize the targets.
μtarget, σtarget = rescale!(train_target; obsdim = 1)
rescale!(test_target, μtarget, σtarget; obsdim = 1);
nothing # hide
```

Now we construct our model

```@example 1
@model function linear_regression(x, y)
    # Set variance prior.
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)

    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))

    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(10))

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, sqrt(σ₂))
end

model = linear_regression(train, train_target)
```

Pathfinder expects the parameters to be in an unconstrained space, so for now we need to get into some Turing internals.

```@example 1
rng = MersenneTwister(42)
sampler = NUTS()
spl = DynamicPPL.Sampler(sampler)
vi = DynamicPPL.VarInfo(rng, model)
DynamicPPL.link!(vi, spl)
nparams = length(vi[spl])
model(rng, vi, spl)

logπ = Turing.Inference.gen_logπ(vi, spl, model)
∂logπ∂θ = last ∘ Turing.Inference.gen_∂logπ∂θ(vi, spl, model)
```

Now we have what we need to run Pathfinder.
We'll initialize with ``\theta_0 \sim \operatorname{Uniform}(-2, 2)`` as recommended by the Pathfinder paper.


```@example 1
θ₀ = rand(rng, nparams) .* 4 .- 2
θ = pathfinder(logπ, ∂logπ∂θ, θ₀, 1; rng)[2][:, 1]
```

Then we transform the parameters back to the original (constrained) parameter space.

```@example 1
vi[spl] = θ
DynamicPPL.invlink!(vi, spl)
x₀ = vi[spl]
```

Finally, we use this as the initial parameters for sampling.

```@example 1
chns = sample(rng, model, sampler, 3_000; init_params=x₀)
```
