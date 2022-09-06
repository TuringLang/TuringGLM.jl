# TuringGLM

Documentation for [TuringGLM](https://github.com/TuringLang/TuringGLM.jl).
Please [file an issue](https://github.com/TuringLang/TuringGLM.jl/issues/new)
if you run into any problems.

## Getting Started

TuringGLM makes easy to specify Bayesian **G**eneralized **L**inear **M**odels using the formula syntax and returns an instantiated [Turing](https://github.com/TuringLang/Turing.jl) model.

Heavily inspired by [brms](https://github.com/paul-buerkner/brms/) (uses RStan or CmdStanR) and [bambi](https://github.com/bambinos/bambi) (uses PyMC3).

### `@formula`

The `@formula` macro is extended from [`StatsModels.jl`](https://github.com/JuliaStats/StatsModels.jl) along with  [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl) for the random-effects (a.k.a. group-level predictors).

The syntax is done by using the `@formula` macro and then specifying the dependent variable followed by a tilde `~` then the independent variables separated by a plus sign `+`.

Example:

```julia
@formula(y ~ x1 + x2 + x3)
```

Moderations/interactions can be specified with the asterisk sign `*`, e.g. `x1 * x2`.
This will be expanded to `x1 + x2 + x1:x2`, which, following the principle of hierarchy,
the main effects must also be added along with the interaction effects. Here `x1:x2`
means that the values of `x1` will be multiplied (interacted) with the values of `x2`.

Random-effects (a.k.a. group-level effects) can be specified with the `(term | group)` inside
the `@formula`, where `term` is the independent variable and `group` is the **categorical**
representation (i.e., either a column of `String`s or a `CategoricalArray` in `data`).
You can specify a random-intercept with `(1 | group)`.

Example:

```julia
@formula(y ~ (1 | group) + x1)
```

### Data

TuringGLM supports any `Tables.jl`-compatible data interface.
The most popular ones are `DataFrame`s and `NamedTuple`s.

### Supported Models

TuringGLM supports non-hierarchical and hierarchical models.
For hierarchical models, only single random-intercept hierarchical models are supported.

Currently, for likelihoods `TuringGLM.jl` supports:

* `Normal` (the default if not specified): linear regression
* `TDist`: robust linear regression
* `Bernoulli`: logistic regression
* `Poisson`: Poisson count data regression
* `NegativeBinomial`: negative binomial robust count data regression

## Tutorials

Take a look at the tutorials for all supported likelihood and models.

```@contents
Pages = [
    "tutorials/linear_regression.md",
    "tutorials/logistic_regression.md",
    "tutorials/poisson_regression.md",
    "tutorials/negativebinomial_regression.md",
    "tutorials/robust_regression.md",
    "tutorials/hierarchical_models.md",
    "tutorials/custom_priors.md"
]
Depth = 1
```
