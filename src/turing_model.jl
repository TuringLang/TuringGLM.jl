"""
    turing_model(formula, data; model=Gaussian, priors=DefaultPrior(), standardize=false)

Create a Turing model using `formula` syntax and a `data` source.

# `formula`

`formula` is the the same friendly interface to specify used to specify statistical models by
[`brms`](https://paul-buerkner.github.io/brms/),
[`rstarnarm`](https://mc-stan.org/rstanarm/index.html),
[`bambi`](https://bambinos.github.io/bambi),
[`StatsModels.jl`](https://juliastats.org/StatsModels.jl/latest/) and
[`MixedModels.jl`](https://juliastats.org/MixedModels.jl/dev/).
The syntax is done by using the `@formula` macro and then specifying the dependent variable
followed by a tilde `~` then the indepedent variables separated by a plus sign `+`.

Example: `@formula(y ~ x1 + x2 + x3)`.

Moderations/interactions can be specified with the asterisk sign `*`, e.g. `x1 * x2`.
This will be expanded to `x1 + x2 + x1:x2`, which, following the principle of hierarchy,
the main effects must also be added along with the interaction effects. Here `x1:x2`
means that the values of `x1` will be multiplied (interacted) with the values of `x2`.

TODO: add random-effects.

# `data`

`data` can be any `Tables.jl`-compatible data interface. The most popular ones are DataFrames
and NamedTuples.

# `model`

`model` represents the likelihood function which you want to condition your data on.
Currently, `TuringGLM.jl` supports:

* `Gaussian()` (the default): linear regression
* `Student()`: robust linear regression
* `Logistic()`: logistic regression
* `Pois()`: Poisson count data regression
* `NegBin()`: negative binomial robust count data regression

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
    data::D;
    model=Gaussian(),
    priors=DefaultPrior(),
    standardize=false,
) where {D}

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
    end

    # Random-Effects Conditionals
    if has_ranef(formula)
        # TODO
    else
        if priors isa DefaultPrior
            custom_prior = CustomPrior(
                TDist(3), LocationScale(median(y), mad(y), TDist(3)), nothing
            )
        else
            custom_prior = priors
        end
        @model function normal_model(
            y,
            X;
            predictors=size(X, 2),
            μ_X=μ_X,
            σ_X=σ_X,
            prior=custom_prior,
            residual=1 / std(y),
        )
            α ~ prior.intercept
            β ~ filldist(prior.predictors, predictors)
            σ ~ Exponential(residual)
            y ~ MvNormal(α .+ X * β, σ^2 * I)
            return (; α, β, σ, y)
        end
        return normal_model(y, X)
    end
end

turing_model(
    formula::FormulaTerm,
    data::D,
    model::Gaussian;
    priors::Prior = DefaultPrior(),
    standardize::Bool = false,
   ) where {D} = turing_model(formula, data; priors, standardize)

function turing_model(
    formula::FormulaTerm, data::D, model::Student; priors=DefaultPrior(), standardize=false
) where {D}
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
    end

    # Random-Effects Conditionals
    if has_ranef(formula)
        # Likelihood Conditionals
        # TODO
    else
        if priors isa DefaultPrior
            custom_prior = CustomPrior(
                TDist(3), LocationScale(median(y), mad(y), TDist(3)), Gamma(2, 0.1)
            )
        else
            custom_prior = priors
        end
        @model function student_model(
            y,
            X;
            predictors=size(X, 2),
            μ_X=μ_X,
            σ_X=σ_X,
            prior=custom_prior,
            residual=1 / std(y),
        )
            α ~ prior.intercept
            β ~ filldist(prior.predictors, predictors)
            σ ~ Exponential(residual)
            ν ~ prior.auxiliary
            y ~ arraydist(LocationScale.(α .+ X * β, σ, TDist.(ν)))
            return (; α, β, σ, ν, y)
        end
        return student_model(y, X)
    end
end

function turing_model(
    formula::FormulaTerm, data::D, model::Logistic; priors=DefaultPrior(), standardize=false
) where {D}
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
    end

    # Random-Effects Conditionals
    if has_ranef(formula)
        # TODO
    else
        if priors isa DefaultPrior
            custom_prior = CustomPrior(TDist(3), LocationScale(0, 2.5, TDist(3)), nothing)
        else
            custom_prior = priors
        end
        @model function bernoulli_model(
            y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=custom_prior
        )
            α ~ prior.intercept
            β ~ filldist(prior.predictors, predictors)
            y ~ arraydist(LazyArray(@~ BernoulliLogit.(α .+ X * β)))
            return (; α, β, y)
        end
        return bernoulli_model(y, X)
    end
end
function turing_model(
    formula::FormulaTerm, data::D, model::Pois; priors=DefaultPrior(), standardize=false
) where {D}
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
    end

    # Random-Effects Conditionals
    if has_ranef(formula)
        # TODO
    else
        if priors isa DefaultPrior
            custom_prior = CustomPrior(
                TDist(3), LocationScale(median(y), mad(y), TDist(3)), nothing
            )
        else
            custom_prior = priors
        end
        @model function poisson_model(
            y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=custom_prior
        )
            α ~ prior.intercept
            β ~ filldist(prior.predictors, predictors)
            y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
            return (; α, β, y)
        end
        return poisson_model(y, X; predictors=size(X, 2))
    end
end

function turing_model(
    formula::FormulaTerm, data::D, model::NegBin; priors=DefaultPrior(), standardize=false
) where {D}
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
    end

    # Random-Effects Conditionals
    if has_ranef(formula)
        # TODO
    else
        if priors isa DefaultPrior
            custom_prior = CustomPrior(
                TDist(3), LocationScale(median(y), mad(y), TDist(3)), Gamma(0.01, 0.01)
            )
        else
            custom_prior = priors
        end
        @model function negbin_model(
            y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=custom_prior
        )
            α ~ prior.intercept
            β ~ filldist(prior.predictors, predictors)
            ϕ⁻ ~ prior.auxiliary
            ϕ = 1 / ϕ⁻
            y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
            return (; α, β, ϕ, y)
        end
        return negbin_model(y, X)
    end
end

function turing_model(
    formula::FormulaTerm,
    data::D,
    model::Union{UnivariateDistribution,Model},
    priors=DefaultPrior(),
    standardize=false,
) where {D}
    return throw(
        ArgumentError(
            "Could not find $(model) likelihood. Please check the documentation for supported likelihoods.",
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
