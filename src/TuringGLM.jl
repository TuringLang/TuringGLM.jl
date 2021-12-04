module TuringGLM

using Statistics: mean, std
using StatsModels: ConstantTerm, FormulaTerm, FunctionTerm, Term
using StatsModels: hasintercept, response
using MixedModels: MixedModel, RandomEffectsTerm
using MixedModels: zerocorr

# Different modelmatrix
using MixedModels: MixedModels
using StatsModels: StatsModels

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
