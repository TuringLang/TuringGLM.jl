# 3.0.0

## Breaking changes

TuringGLM now uses [FlexiChains](https://github.com/penelopeysm/FlexiChains.jl) for sampling output, following Turing v0.45's switch away from MCMCChains.

Sampling a model returned by `turing_model` now produces a FlexiChains chain rather than an `MCMCChains.Chains`. Post-processing that indexed the old chain type (for example `names(chn, :parameters)` or `chn[:, :param, :]`) needs to move to the FlexiChains API.

MCMCChains is still supported. Load it yourself and pass `chain_type=MCMCChains.Chains` to `sample`. It is no longer a dependency, so importing it through TuringGLM or Turing no longer works.

`summarystats` and `quantile`, along with the FlexiChains helpers `At`, `VNChain`, and `SymChain`, are now re-exported from FlexiChains.

The minimum supported Turing version is now 0.45, with the compat range set to `0.45 - 0.46`.
