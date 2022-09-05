"""
    turing_model(formula, data; model=Normal, priors=DefaultPrior(), standardize=false)

Create a Turing model using `formula` syntax and a `data` source.

# `formula`

`formula` is the the same friendly interface to specify used to specify statistical models by
[`brms`](https://paul-buerkner.github.io/brms/),
[`rstarnarm`](https://mc-stan.org/rstanarm/index.html),
[`bambi`](https://bambinos.github.io/bambi),
[`StatsModels.jl`](https://juliastats.org/StatsModels.jl/latest/) and
[`MixedModels.jl`](https://juliastats.org/MixedModels.jl/dev/).
The syntax is done by using the `@formula` macro and then specifying the dependent variable
followed by a tilde `~` then the independent variables separated by a plus sign `+`.

Example: `@formula(y ~ x1 + x2 + x3)`.

Moderations/interactions can be specified with the asterisk sign `*`, e.g. `x1 * x2`.
This will be expanded to `x1 + x2 + x1:x2`, which, following the principle of hierarchy,
the main effects must also be added along with the interaction effects. Here `x1:x2`
means that the values of `x1` will be multiplied (interacted) with the values of `x2`.

Random-effects (a.k.a. group-level effects) can be specified with the `(term | group)` inside
the `@formula`, where `term` is the independent variable and `group` is the **categorical**
representation (i.e., either a column of `String`s or a `CategoricalArray` in `data`).
You can specify a random-intercept with `(1 | group)`.

Example: `@formula(y ~ (1 | group) + x1)`.

**Notice: random-effects are currently only implemented for a single group-level intercept.
Future versions of `TuringGLM.jl` will support slope random-effects and multiple group-level
effets.**

# `data`

`data` can be any `Tables.jl`-compatible data interface.
The most popular ones are `DataFrame`s and `NamedTuple`s.

# `model`

`model` represents the likelihood function which you want to condition your data on.
It has to be a subtype of `Distributions.UnivariateDistribution`.
Currently, `TuringGLM.jl` supports:

* `Normal` (the default if not specified): linear regression
* `TDist`: robust linear regression
* `Bernoulli`: logistic regression
* `Poisson`: Poisson count data regression
* `NegativeBinomial`: negative binomial robust count data regression

# `priors`

`TuringGLM.jl` comes with state-of-the-art default priors, based on the literature and the
Stan community.
By default, `turing_model` will use `DefaultPrior`. But you can specify your own with
`priors=CustomPrior(predictors, intercept, auxiliary)`. All models take a `predictors` and
`intercept` priors.

In robust models, e.g. Linear Regression with Student-t likelihood or Count Regression
with Negative Binomial likelihood, often there is an extra auxiliary parameter that is
needed to parametrize to model to overcome under- or over-dispersion. If you are specifying
a custom prior for one of these type of models, then you should also specify a prior for
the auxiliary parameter.

Non-robust models do not need an auxiliary parameter and you can pass `nothing` as the
auxiliary argument.

Example for a non-robust model: `@formula(y, ...), data; priors=CustomPrior(Normal(0, 2.5), Normal(10, 5), nothing)`

Example for a robust model: `@formula(y, ...), data; priors=CustomPrior(Normal(0, 2.5), Normal(10, 5), Exponential(1))`

# `standardize`

Whether `true` or `false` to standardize your data to mean 0 and standard deviation 1
before inference. Some science fields prefer to analyze and report effects in terms of
standard devations. Also, whenever measurement scales differs, it is often suggested to
standardize the effects for better comparison. By default, `turing_model` sets `standardize=false`.
"""
function turing_model(
    formula::FormulaTerm,
    data;
    model::Type{T}=Normal,
    priors::Prior=DefaultPrior(),
    standardize::Bool=false,
) where {T<:UnivariateDistribution}
    return _turing_model(formula, data, T; priors, standardize)
end

function _turing_model(
    formula::FormulaTerm,
    data,
    ::Type{T};
    priors::Prior=DefaultPrior(),
    standardize::Bool=false,
) where {T<:UnivariateDistribution}
    # extract y, X and Z
    y = data_response(formula, data)
    X = data_fixed_effects(formula, data)
    Z = data_random_effects(formula, data)

    # μ and σ identities
    μ_X = 0
    σ_X = 1
    μ_y = 0
    σ_y = 1

    if standardize
        μ_X, σ_X, X = standardize_predictors(X)
        μ_y, σ_y, y = standardize_predictors(y)
        if !isnothing(Z)
            #TODO: implement random-effects slope
            throw(
                ArgumentError(
                    "TuringGLM currently does not support random-effects for slope terms"
                ),
            )
        end
    end

    # Random-Effects Conditionals
    prior = _prior(priors, y, T)
    ranef = TuringGLM.ranef(formula)
    model = if ranef === nothing
        _model(μ_X, σ_X, prior, T)
    else
        intercept_ranef = intercept_per_ranef(ranef)
        group_var = first(ranef).rhs
        idx = get_idx(term(group_var), data)
        # print for the user the idx
        println("The idx are $(last(idx))\n")
        _model(μ_X, σ_X, prior, intercept_ranef, idx, T)
    end
    return model(y, X)
end

# Default priors
_prior(prior::Prior, y, ::Type{<:UnivariateDistribution}) = prior
function _prior(::DefaultPrior, y, ::Type{Normal})
    m = median(y)
    return CustomPrior(TDist(3), m + mad(y; center=m, normalize=true) * TDist(3), nothing)
end
function _prior(::DefaultPrior, y, ::Type{TDist})
    m = median(y)
    return CustomPrior(
        TDist(3), m + mad(y; center=m, normalize=true) * TDist(3), Gamma(2, 0.1)
    )
end
function _prior(::DefaultPrior, y, ::Type{Bernoulli})
    return CustomPrior(TDist(3), 2.5 * TDist(3), nothing)
end
function _prior(::DefaultPrior, y, ::Type{Poisson})
    return CustomPrior(TDist(3), 2.5 * TDist(3), nothing)
end
function _prior(::DefaultPrior, y, ::Type{NegativeBinomial})
    return CustomPrior(TDist(3), 2.5 * TDist(3), Gamma(0.01, 0.01))
end
function _prior(::DefaultPrior, y, T::Type{<:UnivariateDistribution})
    return throw(
        ArgumentError(
            "No default prior implemented for likelihood type $T. Please check the documentation for supported likelihoods.",
        ),
    )
end

# Models with Normal likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{Normal})
    idxs = first(idx)
    n_gr = length(unique(first(idx)))
    @model function normal_model_ranef(
        y,
        X;
        predictors=size(X, 2),
        idxs=idxs,
        n_gr=n_gr,
        intercept_ranef=intercept_ranef,
        μ_X=μ_X,
        σ_X=σ_X,
        prior=prior,
        residual=1 / std(y),
        mad_y=mad(y; normalize=true),
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(residual)
        μ = α .+ X * β
        if !isempty(intercept_ranef)
            τ ~ mad_y * truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            αⱼ = zⱼ .* τ
            μ .+= αⱼ[idxs]
        end
        #TODO: implement random-effects slope
        y ~ MvNormal(μ, σ^2 * I)
        return (; α, β, σ, τ, zⱼ, αⱼ, y)
    end
end
function _model(μ_X, σ_X, prior, ::Type{Normal})
    @model function normal_model(
        y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior, residual=1 / std(y)
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(residual)
        y ~ MvNormal(α .+ X * β, σ^2 * I)
        return (; α, β, σ, y)
    end
end

# Models with Student-t likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{TDist})
    @model function student_model_ranef(
        y,
        X;
        predictors=size(X, 2),
        idxs=first(idx),
        n_gr=length(unique(first(idx))),
        intercept_ranef=intercept_ranef,
        μ_X=μ_X,
        σ_X=σ_X,
        prior=prior,
        residual=1 / std(y),
        mad_y=mad(y; normalize=true),
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(residual)
        ν ~ prior.auxiliary
        μ = α .+ X * β
        if !isempty(intercept_ranef)
            τ ~ mad_y * truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            αⱼ = zⱼ .* τ
            μ .+= αⱼ[idxs]
        end
        #TODO: implement random-effects slope
        y ~ arraydist(μ + σ * TDist.(ν))
        return (; α, β, σ, ν, τ, zⱼ, αⱼ, y)
    end
end
function _model(μ_X, σ_X, prior, ::Type{TDist})
    @model function student_model(
        y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior, residual=1 / std(y)
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(residual)
        ν ~ prior.auxiliary
        y ~ arraydist((α .+ X * β) .+ σ .* TDist.(ν))
        return (; α, β, σ, ν, y)
    end
end

# Models with Bernoulli likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{Bernoulli})
    @model function bernoulli_model_ranef(
        y,
        X;
        predictors=size(X, 2),
        idxs=first(idx),
        n_gr=length(unique(first(idx))),
        intercept_ranef=intercept_ranef,
        μ_X=μ_X,
        σ_X=σ_X,
        prior=prior,
        mad_y=mad(y; normalize=true),
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        μ = α .+ X * β
        if !isempty(intercept_ranef)
            τ ~ mad_y * truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            αⱼ = zⱼ .* τ
            μ .+= αⱼ[idxs]
        end
        #TODO: implement random-effects slope
        y ~ arraydist(LazyArray(@~ BernoulliLogit.(μ)))
        return (; α, β, τ, zⱼ, αⱼ, y)
    end
end
function _model(μ_X, σ_X, prior, ::Type{Bernoulli})
    @model function bernoulli_model(
        y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        y ~ arraydist(LazyArray(@~ BernoulliLogit.(α .+ X * β)))
        return (; α, β, y)
    end
end

# Models with Poisson likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{Poisson})
    @model function poisson_model_ranef(
        y,
        X;
        predictors=size(X, 2),
        idxs=first(idx),
        n_gr=length(unique(first(idx))),
        intercept_ranef=intercept_ranef,
        μ_X=μ_X,
        σ_X=σ_X,
        prior=prior,
        mad_y=mad(y; normalize=true),
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        μ = α .+ X * β
        if !isempty(intercept_ranef)
            τ ~ mad_y * truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            αⱼ = zⱼ .* τ
            μ .+= αⱼ[idxs]
        end
        #TODO: implement random-effects slope
        y ~ arraydist(LazyArray(@~ LogPoisson.(μ)))
        return (; α, β, τ, zⱼ, αⱼ, y)
    end
end
function _model(μ_X, σ_X, prior, ::Type{Poisson})
    @model function poisson_model(
        y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
        return (; α, β, y)
    end
end

# Models with NegativeBinomial likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{NegativeBinomial})
    @model function negbin_model_ranef(
        y,
        X;
        predictors=size(X, 2),
        idxs=first(idx),
        n_gr=length(unique(first(idx))),
        intercept_ranef=intercept_ranef,
        μ_X=μ_X,
        σ_X=σ_X,
        prior=prior,
        mad_y=mad(y; normalize=true),
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        ϕ⁻ ~ prior.auxiliary
        ϕ = 1 / ϕ⁻
        μ = α .+ X * β
        if !isempty(intercept_ranef)
            τ ~ mad_y * truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            αⱼ = zⱼ .* τ
            μ .+= αⱼ[idxs]
        end
        #TODO: implement random-effects slope
        y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(μ), ϕ)))
        return (; α, β, ϕ, τ, zⱼ, αⱼ, y)
    end
end
function _model(μ_X, σ_X, prior, ::Type{NegativeBinomial})
    @model function negbin_model(y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior)
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        ϕ⁻ ~ prior.auxiliary
        ϕ = 1 / ϕ⁻
        y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
        return (; α, β, ϕ, y)
    end
end

# Unsupported models
function _model(μ_X, σ_X, prior, intercept_ranef, idx, T::Type{<:UnivariateDistribution})
    return throw(
        ArgumentError(
            "No Turing model implemented for likelihood type $T. Please check the documentation for supported likelihoods.",
        ),
    )
end
function _model(μ_X, σ_X, prior, T::Type{<:UnivariateDistribution})
    return throw(
        ArgumentError(
            "No Turing model implemented for likelihood type $T. Please check the documentation for supported likelihoods.",
        ),
    )
end

"""
    NegativeBinomial2(μ, ϕ)

An alternative parameterization of the Negative Binomial distribution:

```math
\\text{Negative-Binomial}(n \\mid \\mu, \\phi) \\sim \\binom{n + \\phi - 1}{n} \\left( \\frac{\\mu}{\\mu + \\phi} \\right)^{n!} \\left( \\frac{\\phi}{\\mu + \\phi} \\right)^{\\phi!}
```

where the expectation is μ and variance is (μ + μ²/ϕ).

The alternative parameterization is inspired by the [Stan's `neg_binomial_2` function](https://mc-stan.org/docs/functions-reference/nbalt.html).
"""
function NegativeBinomial2(μ::T, ϕ::T) where {T<:Real}
    p = max(1 / (1 + μ / ϕ), 1e-6) # numerical stability
    r = ϕ
    return NegativeBinomial(r, p)
end
