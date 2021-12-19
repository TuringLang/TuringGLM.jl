abstract type Prior end

struct DefaultPrior <: Prior end

"""
    CustomPrior(predictors, intercept, auxiliary)

struct to hold information regarding user-specified custom priors.

# Usage

The `CustomPrior` struct has 3 fields:

1. `predictors`: the β coefficients.
2. `intercept`: the α intercept.
3. `auxiliary`: an auxiliary parameter.

In robust models, e.g. Linear Regression with Student-t likelihood or Count Regression
with Negative Binomial likelihood, often there is an extra auxiliary parameter that is
needed to parametrize to model to overcome under- or over-dispersion. If you are specifying
a custom prior for one of these type of models, then you should also specify a prior for
the auxiliary parameter.

Non-robust models do not need an auxiliary parameter and you can pass `nothing` as the
auxiliary argument.
"""
struct CustomPrior <: Prior
    predictors
    intercept
    auxiliary
end

function CustomPrior()
    throw(ArgumentError("CustomPrior has no default parameters, please use DefaultPrior"))
    return nothing
end
