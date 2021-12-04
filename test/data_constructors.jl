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

        f = @formula y_float ~ x_int + (1 + x_float | x_cat)
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

        f = @formula y_float ~ x_int + (1 + x_float | x_cat)
        @test T.n_ranef(f) == 2

        f = @formula y_float ~ 1 + (1 + x_int + x_float | x_cat)
        @test T.n_ranef(f) == 3

        f = @formula y_float ~
            1 + (1 + x_int + x_float | x_cat) + (1 + x_int + x_float | group)
        @test T.n_ranef(f) == 6
    end

    @testset "intercept_per_ranef" begin
        f = @formula y_float ~ 1 + x_int + xcat + (1 | x_cat)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat"]

        f = @formula y_float ~ 1 + (1 + x_int + x_float | x_cat)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat"]

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat"]

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | group)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat", "group"]

        f = @formula y_float ~ 1 + (1 + x_int + x_float | x_cat)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat"]

        f = @formula y_float ~
            1 + (1 + x_int + x_float | x_cat) + (1 + x_int + x_float | group)
        @test T.intercept_per_ranef(T.ranef(f)) == ["x_cat", "group"]
    end

    @testset "slope_per_ranef" begin
        f = @formula y_float ~ 1 + x_int + xcat + (1 | x_cat)
        @test T.slope_per_ranef(T.ranef(f)) == T.SlopePerRanEf()

        f = @formula y_float ~ 1 + (1 + x_int + x_float | x_cat)
        @test T.slope_per_ranef(T.ranef(f)) ==
            T.SlopePerRanEf(Dict("x_cat" => ["x_int", "x_float"]))

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat)
        @test T.slope_per_ranef(T.ranef(f)) == T.SlopePerRanEf()

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | group)
        @test T.slope_per_ranef(T.ranef(f)) == T.SlopePerRanEf()

        f = @formula y_float ~
            1 + (1 + x_int + x_float | x_cat) + (1 + x_int + x_float | group)
        @test T.slope_per_ranef(T.ranef(f)) == T.SlopePerRanEf(
            Dict("x_cat" => ["x_int", "x_float"], "group" => ["x_int", "x_float"])
        )
    end

    @testset "has_zerocorr" begin
        f = @formula y_float ~ 1 + x_int + (1 | x_cat)
        @test T.has_zerocorr(f) == false

        f = @formula y_float ~ 0 + x_int + (1 | x_cat)
        @test T.has_zerocorr(f) == false

        f = @formula y_float ~ x_int + (1 | x_cat)
        @test T.has_zerocorr(f) == false

        f = @formula y_float ~ 1 + x_int + (1 | x_cat) + zerocorr(x_cat)
        @test T.has_zerocorr(f) == true

        f = @formula y_float ~ 0 + x_int + (1 | x_cat) + zerocorr(x_cat)
        @test T.has_zerocorr(f) == true

        f = @formula y_float ~ x_int + (1 | x_cat) + zerocorr(x_cat)
        @test T.has_zerocorr(f) == true

        f = @formula y_float ~ 1 + x_float + (1 | x_cat) + (1 | x_cat) + zerocorr(x_cat)
        @test T.has_zerocorr(f) == true

        f = @formula y_float ~
            1 +
            (1 + x_int + x_float | x_cat) +
            (1 + x_int + x_float | group) +
            zerocorr(group)
        @test T.has_zerocorr(f) == true
    end
end
