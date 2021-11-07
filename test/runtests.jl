using TuringGLM
using Test
using CategoricalArrays: categorical

df = (;
    x_float=[1.1, 2.3, 3.14, 3.65],
    x_int=[1, 2, 3, 4],
    x_cat=categorical([1, 2, 3, 4]),
    x_cat_ordered=categorical([1, 2, 3, 4]; ordered=true),
    y_float=[2.3, 3.4, 4.5, 5.4],
    y_int=[2, 3, 4, 5],
)

my_tests = []

@testset "TuringGLM.jl" begin
    for test in my_tests
        include(test)
    end
end
