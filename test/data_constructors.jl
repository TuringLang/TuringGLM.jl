@testset "data_constructors.jl" begin
    @testset "data_response" begin
        @testset "NamedTuples" begin
            f = @formula y_int ~ 0 + x_float + x_cat
            y = TuringGLM.data_response(f, nt_str)
            @test y == [2, 3, 4, 5]
            y = TuringGLM.data_response(f, nt_cat)
            @test y == [2, 3, 4, 5]
            f = @formula y_int ~ 1 + x_float + x_cat
            y = TuringGLM.data_response(f, nt_str)
            @test y == [2, 3, 4, 5]
            y = TuringGLM.data_response(f, nt_cat)
            @test y == [2, 3, 4, 5]

            f = @formula y_float ~ 0 + x_float + x_cat
            y = TuringGLM.data_response(f, nt_str)
            @test y == [2.3, 3.4, 4.5, 5.4]
            y = TuringGLM.data_response(f, nt_cat)
            @test y == [2.3, 3.4, 4.5, 5.4]
            f = @formula y_float ~ 1 + x_float + x_cat
            y = TuringGLM.data_response(f, nt_str)
            @test y == [2.3, 3.4, 4.5, 5.4]
            y = TuringGLM.data_response(f, nt_cat)
            @test y == [2.3, 3.4, 4.5, 5.4]
        end
        @testset "DataFrames" begin
            f = @formula y_int ~ 0 + x_float + x_cat
            y = TuringGLM.data_response(f, df_str)
            @test y == [2, 3, 4, 5]
            y = TuringGLM.data_response(f, df_cat)
            @test y == [2, 3, 4, 5]
            f = @formula y_int ~ 1 + x_float + x_cat
            y = TuringGLM.data_response(f, df_str)
            @test y == [2, 3, 4, 5]
            y = TuringGLM.data_response(f, df_cat)
            @test y == [2, 3, 4, 5]

            f = @formula y_float ~ 0 + x_float + x_cat
            y = TuringGLM.data_response(f, df_str)
            @test y == [2.3, 3.4, 4.5, 5.4]
            y = TuringGLM.data_response(f, df_cat)
            @test y == [2.3, 3.4, 4.5, 5.4]
            f = @formula y_float ~ 1 + x_float + x_cat
            y = TuringGLM.data_response(f, df_str)
            @test y == [2.3, 3.4, 4.5, 5.4]
            y = TuringGLM.data_response(f, df_cat)
            @test y == [2.3, 3.4, 4.5, 5.4]
        end
    end
    @testset "data_fixed_effects" begin
        @testset "NamedTuples" begin
            f = @formula y_int ~ 0 + x_float + x_cat
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula y_int ~ 1 + x_float + x_cat
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula y_float ~ 0 + x_float + x_cat
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula y_float ~ 1 + x_float + x_cat
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula y_int ~ 0 + x_float + x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula y_int ~ 1 + x_float + x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula y_float ~ 0 + x_float + x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula y_float ~ 1 + x_float + x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            # Interactions
            f = @formula y_float ~ 0 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 1 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 0 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 1 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 0 + x_int * x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 1 + x_int * x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, nt_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]

            # Interactions coming first
            f = @formula y_float ~ 1 + x_int * x_cat + x_float
            X = TuringGLM.data_fixed_effects(f, nt_str)
            @test X == [
                1.0 0.0 0.0 0.0 1.1 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.3 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 3.14 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 3.65 0.0 0.0 4.0
            ]
        end
        @testset "DataFrames" begin
            f = @formula(y_int ~ 0 + x_float + x_cat)
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula(y_int ~ 1 + x_float + x_cat)
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula(y_float ~ 0 + x_float + x_cat)
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula(y_float ~ 1 + x_float + x_cat)
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula(y_int ~ 0 + x_float + x_cat_ordered)
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula(y_int ~ 1 + x_float + x_cat_ordered)
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula(y_float ~ 0 + x_float + x_cat_ordered)
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]
            f = @formula(y_float ~ 1 + x_float + x_cat_ordered)
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            # Interactions
            f = @formula y_float ~ 0 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 1 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 0 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 1 + x_int * x_cat
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 0 + x_int * x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]
            f = @formula y_float ~ 1 + x_int * x_cat_ordered
            X = TuringGLM.data_fixed_effects(f, df_cat)
            @test X == [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]

            # Interactions coming first
            f = @formula y_float ~ 1 + x_int * x_cat + x_float
            X = TuringGLM.data_fixed_effects(f, df_str)
            @test X == [
                1.0 0.0 0.0 0.0 1.1 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.3 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 3.14 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 3.65 0.0 0.0 4.0
            ]
        end
    end
    @testset "data_random_effects" begin end
end
