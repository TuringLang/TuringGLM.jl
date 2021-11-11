@testset "data_constructors.jl" begin
    @testset "data_response" begin
        # NamedTuples
        f = @formula(y_int ~ x_float + x_cat)
        y = TuringGLM.data_response(f, nt_str)
        @test y == [2, 3, 4, 5]
        y = TuringGLM.data_response(f, nt_cat)
        @test y == [2, 3, 4, 5]

        f = @formula(y_float ~ x_float + x_cat)
        y = TuringGLM.data_response(f, nt_str)
        @test y == [2.3, 3.4, 4.5, 5.4]
        y = TuringGLM.data_response(f, nt_cat)
        @test y == [2.3, 3.4, 4.5, 5.4]

        # DataFrames
        f = @formula(y_int ~ x_float + x_cat)
        y = TuringGLM.data_response(f, df_str)
        @test y == [2, 3, 4, 5]
        y = TuringGLM.data_response(f, df_cat)
        @test y == [2, 3, 4, 5]

        f = @formula(y_float ~ x_float + x_cat)
        y = TuringGLM.data_response(f, df_str)
        @test y == [2.3, 3.4, 4.5, 5.4]
        y = TuringGLM.data_response(f, df_cat)
        @test y == [2.3, 3.4, 4.5, 5.4]
    end
    @testset "data_fixed_effects" begin
        # NamedTuples
        f = @formula(y_int ~ x_float + x_cat)
        m = TuringGLM.data_fixed_effects(f, nt_str)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ x_float + x_cat)
        m = TuringGLM.data_fixed_effects(f, nt_str)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_int ~ x_float + x_cat_ordered)
        m = TuringGLM.data_fixed_effects(f, nt_cat)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ x_float + x_cat_ordered)
        m = TuringGLM.data_fixed_effects(f, nt_cat)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        # DataFrames
        f = @formula(y_int ~ x_float + x_cat)
        m = TuringGLM.data_fixed_effects(f, df_str)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ x_float + x_cat)
        m = TuringGLM.data_fixed_effects(f, df_str)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_int ~ x_float + x_cat_ordered)
        m = TuringGLM.data_fixed_effects(f, df_cat)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ x_float + x_cat_ordered)
        m = TuringGLM.data_fixed_effects(f, df_cat)
        @test X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]
    end
    @testset "data_random_effects" begin
    end
end
