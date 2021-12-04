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

using Reexport: @reexport

@reexport begin
    using Turing
    using StatsModels: @formula
    using MixedModels: zerocorr
end

include("utils.jl")
include("data_constructors.jl")

export @formula

end # module
