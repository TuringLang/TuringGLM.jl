using TuringGLM
using Test
using DataFrames
using CategoricalArrays: categorical
using Statistics: mean, std

nt_str = (;
    x_float=[1.1, 2.3, 3.14, 3.65],
    x_int=[1, 2, 3, 4],
    x_cat=["1", "2", "3", "4"],
    y_float=[2.3, 3.4, 4.5, 5.4],
    y_int=[2, 3, 4, 5],
)

nt_cat = (;
    x_float=[1.1, 2.3, 3.14, 3.65],
    x_int=[1, 2, 3, 4],
    x_cat=categorical([1, 2, 3, 4]),
    x_cat_ordered=categorical([1, 2, 3, 4]; ordered=true),
    y_float=[2.3, 3.4, 4.5, 5.4],
    y_int=[2, 3, 4, 5],
)

df_str = DataFrame(;
    x_float=[1.1, 2.3, 3.14, 3.65],
    x_int=[1, 2, 3, 4],
    x_cat=["1", "2", "3", "4"],
    y_float=[2.3, 3.4, 4.5, 5.4],
    y_int=[2, 3, 4, 5],
)

df_cat = DataFrame(;
    x_float=[1.1, 2.3, 3.14, 3.65],
    x_int=[1, 2, 3, 4],
    x_cat=categorical([1, 2, 3, 4]),
    x_cat_ordered=categorical([1, 2, 3, 4]; ordered=true),
    y_float=[2.3, 3.4, 4.5, 5.4],
    y_int=[2, 3, 4, 5],
)

@testset "TuringGLM.jl" begin
    include("formula.jl")
end
