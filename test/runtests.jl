using TuringGLM
using Test
using DataFrames
using CategoricalArrays: categorical
using Statistics: mean, std

x_float = [1.1, 2.3, 3.14, 3.65]
x_int = [1, 2, 3, 4]
y_float = [2.3, 3.4, 4.5, 5.4]
y_int = [2, 3, 4, 5]

nt_str = (; x_float, x_int=[1, 2, 3, 4], x_cat=string.(x_int), y_float, y_int)

nt_cat = (;
    x_float,
    x_int,
    x_cat=categorical(x_int),
    x_cat_ordered=categorical(x_int; ordered=true),
    y_float,
    y_int,
)

df_str = DataFrame(nt_str)

df_cat = DataFrame(nt_cat)

@testset "TuringGLM.jl" begin
    include("formula.jl")
    include("terms.jl")
    include("schema.jl")
    include("error_messages.jl")
    include("contrasts.jl")
    include("data_constructors.jl")
    include("utils.jl")
end
