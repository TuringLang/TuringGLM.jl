module TuringGLM

using Statistics: mean, std

import Tables
import MixedModels
import StatsModels

using Reexport: @reexport

@reexport begin
    using Turing
    using MixedModels: @formula
    # TODO: review which Link types will be used
    #   See GLM.jl/src/glmtools.jl and GLM.jl/src/GLM.jl
end

include("formula.jl")

end # module
