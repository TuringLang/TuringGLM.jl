module TuringGLM

using Reexport: @reexport

@reexport begin
    using Turing
    using StatsModels: @formula
    # TODO: review which Link types will be used
    #   See GLM.jl/src/glmtools.jl and GLM.jl/src/GLM.jl
end

include("formula.jl")

end # module
