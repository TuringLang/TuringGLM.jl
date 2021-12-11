module TuringGLM

import Base: ==, length
using Statistics: mean, std
using StatsModels: ConstantTerm, FormulaTerm, FunctionTerm, Term
using StatsModels: hasintercept, response, term
using MixedModels: MixedModel, RandomEffectsTerm, ZeroCorr
using MixedModels: zerocorr

# Different modelmatrix
using MixedModels: MixedModels
using StatsModels: StatsModels

# Tables API stuff
using Tables: Tables
using TableOperations: TableOperations

# LazyArrays
using LazyArrays: @~, LazyArray

using Reexport: @reexport

@reexport begin
    using Turing
    using StatsModels: @formula
    using MixedModels: zerocorr
    using GLM: # Distributions
        Bernoulli,
        Binomial,
        Gamma,
        InverseGaussian,
        NegativeBinomial,
        Normal,
        Poisson,
        TDist
    using GLM: # Link
        Link,
        CauchitLink,
        CloglogLink,
        IdentityLink,
        InverseLink,
        InverseSquareLink,
        LogitLink,
        LogLink,
        NegativeBinomialLink,
        ProbitLink,
        SqrtLink
    using GLM: canonicallink  # canonical link function for a distribution
end

include("utils.jl")
include("data_constructors.jl")
include("turing_model.jl")

export turing_model

end # module
