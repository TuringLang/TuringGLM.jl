"""
    turing_model(formula, data; family=Normal, priors=DefaultPrior(), standardize=false)

yada Yada
"""
function turing_model(
    formula::FormulaTerm, data::D; family="normal", priors=DefaultPrior(), standardize=false
) where {D}

    # extract y, X and Z
    y = data_response(formula, data)
    X = data_fixed_effects(formula, data)
    Z = data_random_effects(formula, data)

    # μ_X and σ_x identities
    μ_X = 0
    σ_X = 1

    if standardize
        μ_X, σ_X, X = standardize_predictors(X)
    end

    # Random-Effects Conditionals
    if has_ranef(formula)
        # Likelihood Conditionals
        # TODO
    else
        # Likelihood Conditionals
        if family == "normal"
            if priors isa DefaultPrior
                custom_prior = CustomPrior(
                    TDist(3),
                    LocationScale(median(y), mad(y), TDist(3)),
                    nothing,
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
                y ~ MvNormal(α .+ X * β, σ ^ 2 * I)
                return (; α, β, σ, y)
            end
            return normal_model(y, X)
        elseif family == "student"
            if priors isa DefaultPrior
                custom_prior = CustomPrior(
                    TDist(3),
                    LocationScale(median(y), mad(y), TDist(3)),
                    Gamma(2, 0.1),
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
        elseif family == "bernoulli"
            if priors isa DefaultPrior
                custom_prior = CustomPrior(
                    TDist(3),
                    LocationScale(0, 2.5, TDist(3)),
                    nothing,
                )
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
        elseif family == "poisson"
            if priors isa DefaultPrior
                custom_prior = CustomPrior(
                    TDist(3),
                    LocationScale(median(y), mad(y), TDist(3)),
                    nothing,
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
        elseif family == "negativebinomial"
            if priors isa DefaultPrior
                custom_prior = CustomPrior(
                    TDist(3),
                    LocationScale(median(y), mad(y), TDist(3)),
                    Gamma(0.01, 0.01),
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
        else
            throw(
                ArgumentError(
                    "Could not find $(family) likelihood. Please check the documentation for supported likelihoods.",
                ),
            )
        end
    end
end

"""
    NegativeBinomial2(μ, σ)

An alternative parameterization of the Negative Binomial distribution:

```math
\\text{Negative-Binomial}(n \\mid \\mu, \\phi) \\sim \\binom{n + \\phi - 1}{n} \\left( \\frac{\\mu}{\\mu + \\phi} \\right)^{n!} \\left( \\frac{\\phi}{\\mu + \\phi} \\right)^{\\phi!}
```

where the expectation is μ and variance is (μ + μ²/ϕ).

The alternative parameterization is inspired by the [Stan's `neg_binomial_2` function](https://mc-stan.org/docs/functions-reference/nbalt.html).
"""
function NegativeBinomial2(μ::T, ϕ::T) where {T<:Real}
    p = 1 / (1 + μ / ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end
