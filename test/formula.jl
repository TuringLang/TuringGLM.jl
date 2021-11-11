@testset "formula" begin
    @testset "model_response" begin
        # NamedTuples
        f = @formula(y_int ~ 1 + x_float + x_cat)
        y = TuringGLM.model_response(f, nt_str)
        @test y == [2, 3, 4, 5]
        y = TuringGLM.model_response(f, nt_cat)
        @test y == [2, 3, 4, 5]

        f = @formula(y_float ~ 1 + x_float + x_cat)
        y = TuringGLM.model_response(f, nt_str)
        @test y == [2.3, 3.4, 4.5, 5.4]
        y = TuringGLM.model_response(f, nt_cat)
        @test y == [2.3, 3.4, 4.5, 5.4]

        # DataFrames
        f = @formula(y_int ~ 1 + x_float + x_cat)
        y = TuringGLM.model_response(f, df_str)
        @test y == [2, 3, 4, 5]
        y = TuringGLM.model_response(f, df_cat)
        @test y == [2, 3, 4, 5]

        f = @formula(y_float ~ 1 + x_float + x_cat)
        y = TuringGLM.model_response(f, df_str)
        @test y == [2.3, 3.4, 4.5, 5.4]
        y = TuringGLM.model_response(f, df_cat)
        @test y == [2.3, 3.4, 4.5, 5.4]
    end
    @testset "model_matrix" begin
        # NamedTuples
        f = @formula(y_int ~ 1 + x_float + x_cat)
        m = TuringGLM.model_matrix(f, nt_str)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ 1 + x_float + x_cat)
        m = TuringGLM.model_matrix(f, nt_str)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_int ~ 1 + x_float + x_cat_ordered)
        m = TuringGLM.model_matrix(f, nt_cat)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ 1 + x_float + x_cat_ordered)
        m = TuringGLM.model_matrix(f, nt_cat)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        # DataFrames
        f = @formula(y_int ~ 1 + x_float + x_cat)
        m = TuringGLM.model_matrix(f, df_str)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ 1 + x_float + x_cat)
        m = TuringGLM.model_matrix(f, df_str)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_int ~ 1 + x_float + x_cat_ordered)
        m = TuringGLM.model_matrix(f, df_cat)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]

        f = @formula(y_float ~ 1 + x_float + x_cat_ordered)
        m = TuringGLM.model_matrix(f, df_cat)
        @test m.X == [
            1.1 0.0 0.0 0.0
            2.3 1.0 0.0 0.0
            3.14 0.0 1.0 0.0
            3.65 0.0 0.0 1.0
        ]
    end

    @testset "center_predictors" begin
        # NamedTuples
        f = @formula(y_int ~ 1 + x_float + x_cat)
        m = TuringGLM.model_matrix(f, nt_str)
        μ_X, X_centered = TuringGLM.center_predictors(m.X)
        @test μ_X ≈ [2.547 0.25 0.25 0.25] atol = 0.01
        @test X_centered ≈ [
            -1.447 -0.25 -0.25 -0.25
            -0.247 0.75 -0.25 -0.25
            0.592 -0.25 0.75 -0.25
            1.102 -0.25 -0.25 0.75
        ] atol = 0.01
        @test mapslices(mean, X_centered; dims=1) ≈ [0 0 0 0] atol = 0.01

        # DataFrames
        m = TuringGLM.model_matrix(f, df_str)
        μ_X, X_centered = TuringGLM.center_predictors(m.X)
        @test μ_X ≈ [2.547 0.25 0.25 0.25] atol = 0.01
        @test X_centered ≈ [
            -1.447 -0.25 -0.25 -0.25
            -0.247 0.75 -0.25 -0.25
            0.592 -0.25 0.75 -0.25
            1.102 -0.25 -0.25 0.75
        ] atol = 0.01
        @test mapslices(mean, X_centered; dims=1) ≈ [0 0 0 0] atol = 0.01
    end

    @testset "standardize_predictors" begin
        # NamedTuples
        f = @formula(y_int ~ 1 + x_float + x_cat)
        m = TuringGLM.model_matrix(f, nt_str)
        μ_X, σ_X, X_std = TuringGLM.standardize_predictors(m.X)
        @test μ_X ≈ [2.547 0.25 0.25 0.25] atol = 0.01
        @test σ_X ≈ [1.114 0.5 0.5 0.5] atol = 0.01
        @test X_std ≈ [
            -1.299 -0.5 -0.5 -0.5
            -0.222 1.5 -0.5 -0.5
            0.531 -0.5 1.5 -0.5
            0.989 -0.5 -0.5 1.5
        ] atol = 0.01
        @test mapslices(mean, X_std; dims=1) ≈ [0 0 0 0] atol = 0.01
        @test mapslices(std, X_std; dims=1) ≈ [1 1 1 1] atol = 0.01

        # DataFrames
        m = TuringGLM.model_matrix(f, df_str)
        μ_X, σ_X, X_std = TuringGLM.standardize_predictors(m.X)
        @test μ_X ≈ [2.547 0.25 0.25 0.25] atol = 0.01
        @test σ_X ≈ [1.114 0.5 0.5 0.5] atol = 0.01
        @test X_std ≈ [
            -1.299 -0.5 -0.5 -0.5
            -0.222 1.5 -0.5 -0.5
            0.531 -0.5 1.5 -0.5
            0.989 -0.5 -0.5 1.5
        ] atol = 0.01
        @test mapslices(mean, X_std; dims=1) ≈ [0 0 0 0] atol = 0.01
        @test mapslices(std, X_std; dims=1) ≈ [1 1 1 1] atol = 0.01
    end
end
