@testset "data_constructors.jl" begin
    @testset "data_response" begin
        @testset "NamedTuples" begin
            expected = [2, 3, 4, 5]
            f = @formula y_int ~ 0 + x_float + x_cat
            y = T.data_response(f, nt_str)
            @test y == expected
            y = T.data_response(f, nt_cat)
            @test y == expected
            f = @formula y_int ~ 1 + x_float + x_cat
            y = T.data_response(f, nt_str)
            @test y == expected
            y = T.data_response(f, nt_cat)
            @test y == expected

            expected = [2.3, 3.4, 4.5, 5.4]
            f = @formula y_float ~ 0 + x_float + x_cat
            y = T.data_response(f, nt_str)
            @test y == expected
            y = T.data_response(f, nt_cat)
            @test y == expected
            f = @formula y_float ~ 1 + x_float + x_cat
            @test y == expected
            y = T.data_response(f, nt_str)
            @test y == expected
            y = T.data_response(f, nt_cat)
            @test y == expected
        end
        @testset "DataFrames" begin
            expected = [2, 3, 4, 5]
            f = @formula y_int ~ 0 + x_float + x_cat
            y = T.data_response(f, df_str)
            @test y == expected
            y = T.data_response(f, df_cat)
            @test y == expected
            f = @formula y_int ~ 1 + x_float + x_cat
            y = T.data_response(f, df_str)
            @test y == expected
            y = T.data_response(f, df_cat)
            @test y == expected

            expected = [2.3, 3.4, 4.5, 5.4]
            f = @formula y_float ~ 0 + x_float + x_cat
            y = T.data_response(f, df_str)
            @test y == expected
            y = T.data_response(f, df_cat)
            @test y == expected
            f = @formula y_float ~ 1 + x_float + x_cat
            y = T.data_response(f, df_str)
            @test y == expected
            y = T.data_response(f, df_cat)
            @test y == expected
        end
    end
    @testset "data_fixed_effects" begin
        @testset "NamedTuples" begin
            expected = [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula y_int ~ 0 + x_float + x_cat
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected

            f = @formula y_int ~ 1 + x_float + x_cat
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected

            f = @formula y_float ~ 0 + x_float + x_cat
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected

            f = @formula y_float ~ 1 + x_float + x_cat
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected

            f = @formula y_int ~ 0 + x_float + x_cat_ordered
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            f = @formula y_int ~ 1 + x_float + x_cat_ordered
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            f = @formula y_float ~ 0 + x_float + x_cat_ordered
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            f = @formula y_float ~ 1 + x_float + x_cat_ordered
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            # Interactions
            expected = [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]

            f = @formula y_float ~ 0 + x_int * x_cat
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected

            f = @formula y_float ~ 1 + x_int * x_cat
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected

            f = @formula y_float ~ 0 + x_int * x_cat
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            f = @formula y_float ~ 1 + x_int * x_cat
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            f = @formula y_float ~ 0 + x_int * x_cat_ordered
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            f = @formula y_float ~ 1 + x_int * x_cat_ordered
            X = T.data_fixed_effects(f, nt_cat)
            @test X == expected

            # Interactions coming first
            expected = [
                1.0 0.0 0.0 0.0 1.1 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.3 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 3.14 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 3.65 0.0 0.0 4.0
            ]

            f = @formula y_float ~ 1 + x_int * x_cat + x_float
            X = T.data_fixed_effects(f, nt_str)
            @test X == expected
        end
        @testset "DataFrames" begin
            expected = [
                1.1 0.0 0.0 0.0
                2.3 1.0 0.0 0.0
                3.14 0.0 1.0 0.0
                3.65 0.0 0.0 1.0
            ]

            f = @formula(y_int ~ 0 + x_float + x_cat)
            X = T.data_fixed_effects(f, df_str)
            @test X == expected

            f = @formula(y_int ~ 1 + x_float + x_cat)
            X = T.data_fixed_effects(f, df_str)
            @test X == expected

            f = @formula(y_float ~ 0 + x_float + x_cat)
            X = T.data_fixed_effects(f, df_str)
            @test X == expected

            f = @formula(y_float ~ 1 + x_float + x_cat)
            X = T.data_fixed_effects(f, df_str)
            @test X == expected

            f = @formula(y_int ~ 0 + x_float + x_cat_ordered)
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            f = @formula(y_int ~ 1 + x_float + x_cat_ordered)
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            f = @formula(y_float ~ 0 + x_float + x_cat_ordered)
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            f = @formula(y_float ~ 1 + x_float + x_cat_ordered)
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            # Interactions
            expected = [
                1.0 0.0 0.0 0.0 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 0.0 0.0 4.0
            ]

            f = @formula y_float ~ 0 + x_int * x_cat
            X = T.data_fixed_effects(f, df_str)
            @test X == expected

            f = @formula y_float ~ 1 + x_int * x_cat
            X = T.data_fixed_effects(f, df_str)
            @test X == expected

            f = @formula y_float ~ 0 + x_int * x_cat
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            f = @formula y_float ~ 1 + x_int * x_cat
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            f = @formula y_float ~ 0 + x_int * x_cat_ordered
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            f = @formula y_float ~ 1 + x_int * x_cat_ordered
            X = T.data_fixed_effects(f, df_cat)
            @test X == expected

            # Interactions coming first
            expected = [
                1.0 0.0 0.0 0.0 1.1 0.0 0.0 0.0
                2.0 1.0 0.0 0.0 2.3 2.0 0.0 0.0
                3.0 0.0 1.0 0.0 3.14 0.0 3.0 0.0
                4.0 0.0 0.0 1.0 3.65 0.0 0.0 4.0
            ]

            f = @formula y_float ~ 1 + x_int * x_cat + x_float
            X = T.data_fixed_effects(f, df_str)
            @test X == expected
        end
    end
    @testset "data_random_effects" begin end
end
