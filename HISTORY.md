# 3.0.0

## Breaking changes

Sampling now returns a [FlexiChains](https://github.com/penelopeysm/FlexiChains.jl) chain instead of an `MCMCChains.Chains`, following Turing 0.45. Post-processing that relied on MCMCChains indexing, such as `names(chn, :parameters)` or `chn[:, :param, :]`, needs to move to the FlexiChains API.

MCMCChains still works if you load it yourself and pass `chain_type=MCMCChains.Chains` to `sample`. It is no longer a dependency, so you can no longer get it by importing TuringGLM.

`At`, `VNChain`, `SymChain`, `summarystats`, and `quantile` are re-exported from FlexiChains.

## Other changes

The likelihoods use `product_distribution` rather than `arraydist`, which routed through the deprecated `Distributions.Product` constructor and so called `Base.depwarn` on every evaluation. Same distribution type, same results.

Fixed the Student-t model with a random-effects intercept, which threw a `MethodError` instead of sampling.
