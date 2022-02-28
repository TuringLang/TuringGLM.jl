module TuringGLM

import Base: ==, length
using LinearAlgebra: I
using Statistics: mean, median, std
using StatsBase: mad, coefnames
using StatsModels: ConstantTerm, FormulaTerm, FunctionTerm, Term
using StatsModels: hasintercept, response, term
using MixedModels: MixedModel, RandomEffectsTerm

# Different modelmatrix
using MixedModels: MixedModels
using StatsModels: StatsModels

# Tables API stuff
using Tables: Tables
using TableOperations: TableOperations

# LazyArrays
using LazyArrays: @~, LazyArray

# Distributions
using Distributions: UnivariateDistribution

using Reexport: @reexport

@reexport begin
    using Turing
    using StatsModels: @formula
    using Distributions:
        Bernoulli,
        Binomial,
        Gamma,
        Gaussian,
        InverseGaussian,
        NegativeBinomial,
        Normal,
        Poisson,
        TDist
end

include("utils.jl")
include("data_constructors.jl")
include("priors.jl")
include("turing_model.jl")

export turing_model
export CustomPrior, DefaultPrior

end # module
