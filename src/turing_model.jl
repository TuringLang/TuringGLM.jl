"""
    turing_model(formula, data; family=Normal, priors=DefaultPriors(), standardize=false)

yada Yada
"""
function turing_model(
    formula::FormulaTerm,
    data::D;
    family=Normal(),
    # priors=DefaultPriors(),
    standardize=false,
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
        if family isa Normal
            # TODO: Normal likelihood
        elseif family isa TDist
            # TODO: Student likelihood
        elseif family isa Bernoulli
            @model function bernoulli_model(y, X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X)
                α ~ Normal(0, 2.5)
                β ~ filldist(TDist(3), predictors)
                y ~ arraydist(LazyArray(@~ BernoulliLogit.(α .+ X * β)))
                return (; α, β, y)
            end
            return bernoulli_model(y, X)
        elseif family isa Poisson
            @model function poisson_model(y, X; predictors=size(X, 2))
                α ~ Normal(0, 2.5)
                β ~ filldist(TDist(3), predictors)
                y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
                return (; α, β, y)
            end
            return poisson_model(y, X; predictors=size(X, 2))
        elseif family isa NegativeBinomial
            @model function negbin_model(y, X; predictors=size(X, 2))
                α ~ Normal(0, 2.5)
                β ~ filldist(TDist(3), predictors)
                ϕ⁻ ~ Gamma(0.01, 0.01)
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

macro beta(X)
    return :(β ~ filldist(TDist(3), size(X, 2)))
end
