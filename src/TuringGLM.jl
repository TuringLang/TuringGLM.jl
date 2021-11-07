module TuringGLM

using Reexport: @reexport

@reexport begin
    using Turing
    using StatsModels: @formula, hasintercept
    # TODO: review which Link types will be used
    using GLM: # only the Link types
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
end

include("canonical_links.jl")
include("negative_binomial_2.jl")
include("priors.jl")
include("random_effects_terms.jl")
include("turing_model.jl")

end # module
