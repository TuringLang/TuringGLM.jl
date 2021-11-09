module TuringGLM

using Reexport: @reexport

@reexport begin
    using Turing
    using StatsModels: @formula 
    # TODO: review which Link types will be used
    #   See GLM.jl/src/glmtools.jl and GLM.jl/src/GLM.jl
end

include("canonical_links.jl")
include("formula.jl")
include("negative_binomial_2.jl")
include("priors.jl")
include("random_effects_terms.jl")
include("turing_model.jl")

end # module
