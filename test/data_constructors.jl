@timed_testset "data_constructors" begin
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

    @testset "data_random_effects" begin
        expected = nothing
        f = @formula y_float ~ 1 + x_int * x_cat + x_float
        Z = T.data_random_effects(f, nt_str)
        @test Z == expected
        Z = T.data_random_effects(f, nt_cat)
        @test Z == expected
        Z = T.data_random_effects(f, df_str)
        @test Z == expected
        Z = T.data_random_effects(f, df_cat)
        @test Z == expected

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat)
        Z = T.data_random_effects(f, nt_str)
        @test Z == expected
        Z = T.data_random_effects(f, nt_cat)
        @test Z == expected
        Z = T.data_random_effects(f, df_str)
        @test Z == expected
        Z = T.data_random_effects(f, df_cat)
        @test Z == expected
    end

    @testset "has_ranef" begin
        f = @formula y_float ~ 1 + x_int + x_cat
        @test T.has_ranef(f) == false

        f = @formula y_float ~ 1 + x_int + (1 | x_cat)
        @test T.has_ranef(f) == true

        f = @formula y_float ~ 0 + x_int + x_cat
        @test T.has_ranef(f) == false

        f = @formula y_float ~ 0 + x_int + (1 | x_cat)
        @test T.has_ranef(f) == true

        f = @formula y_float ~ x_int + x_cat
        @test T.has_ranef(f) == false

        f = @formula y_float ~ x_int + (1 | x_cat)
        @test T.has_ranef(f) == true
    end

    @testset "ranef" begin
        f = @formula y_float ~ 1 + x_int + x_cat
        @test T.ranef(f) === nothing

        f = @formula y_float ~ x_int + (1 | x_cat)
        @test T.ranef(f) isa Tuple{T.RandomEffectsTerm}
    end

    @testset "n_ranef" begin
        f = @formula y_float ~ 1 + x_int + x_cat
        @test T.n_ranef(f) == 0

        f = @formula y_float ~ x_int + (1 | x_cat)
        @test T.n_ranef(f) == 1

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat)
        @test T.n_ranef(f) == 1

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | group)
        @test T.n_ranef(f) == 2
    end

    @testset "intercept_per_ranef" begin
        f = @formula y_float ~ 1 + x_int + xcat + (1 | x_cat)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat"]

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat"]

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | group)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat", "group"]
    end

    @testset "slope_per_ranef" begin
        f = @formula y_float ~ 1 + x_int + xcat + (1 | x_cat)
        @test T.slope_per_ranef(T.ranef(f)) == T.SlopePerRanEf()

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat)
        @test T.slope_per_ranef(T.ranef(f)) == T.SlopePerRanEf()
    end

    @testset "get_idx" begin
        expected = ([1, 2, 3, 4], Dict(1.1 => 1, 3.65 => 4, 2.3 => 2, 3.14 => 3))
        @test T.get_idx(T.term("x_float"), nt_str) == expected
        @test T.get_idx(T.term("x_float"), df_str) == expected
        @test T.get_idx(T.term("x_float"), nt_cat) == expected
        @test T.get_idx(T.term("x_float"), df_cat) == expected

        expected = ([1, 2, 3, 4], Dict("4" => 4, "1" => 1, "2" => 2, "3" => 3))
        @test T.get_idx(T.term("x_cat"), nt_str) == expected
        @test T.get_idx(T.term("x_cat"), df_str) == expected

        cv = categorical([1, 2, 3, 4])
        expected = (
            [1, 2, 3, 4],
            Dict(
                CategoricalValue(4, cv) => 4,
                CategoricalValue(2, cv) => 2,
                CategoricalValue(3, cv) => 3,
                CategoricalValue(1, cv) => 1,
            ),
        )
        @test T.get_idx(T.term("x_cat"), nt_cat) == expected
        @test T.get_idx(T.term("x_cat"), df_cat) == expected
    end

    @testset "get_var" begin
        expected = [1.1, 2.3, 3.14, 3.65]
        @test T.get_var(T.term("x_float"), nt_str) == expected
        @test T.get_var(T.term("x_float"), df_str) == expected

        expected = [1.1, 2.3, 3.14, 3.65]
        @test T.get_var(T.term("x_float"), nt_cat) == expected
        @test T.get_var(T.term("x_float"), df_cat) == expected

        expected = ["1", "2", "3", "4"]
        @test T.get_var(T.term("x_cat"), nt_str) == expected
        @test T.get_var(T.term("x_cat"), df_str) == expected

        expected = categorical([1, 2, 3, 4])
        @test T.get_var(T.term("x_cat"), nt_cat) == expected
        @test T.get_var(T.term("x_cat"), df_cat) == expected
    end
end
