@testset "formula" begin

    # make_yX
    # Intercept and no intercept must have the same X
    f = @formula(y_int ~ 0 + x_float + x_cat)
    f_intercept = @formula(y_int ~ 1 + x_float + x_cat)
    @test TuringGLM.make_yX(f, nt_str) == TuringGLM.make_yX(f_intercept, nt_str)
    @test TuringGLM.make_yX(f, df_str) == TuringGLM.make_yX(f_intercept, df_str)
    @test TuringGLM.make_yX(f, nt_cat) == TuringGLM.make_yX(f_intercept, nt_cat)
    @test TuringGLM.make_yX(f, df_cat) == TuringGLM.make_yX(f_intercept, df_cat)

    # NamedTuples
    f = @formula(y_int ~ 1 + x_float + x_cat)
    y, X = TuringGLM.make_yX(f, nt_str)
    @test y == [2, 3, 4, 5]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    f = @formula(y_float ~ 1 + x_float + x_cat)
    y, X = TuringGLM.make_yX(f, nt_str)
    @test y == [2.3, 3.4, 4.5, 5.4]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    f = @formula(y_int ~ 1 + x_float + x_cat_ordered)
    y, X = TuringGLM.make_yX(f, nt_cat)
    @test y == [2, 3, 4, 5]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    f = @formula(y_float ~ 1 + x_float + x_cat_ordered)
    y, X = TuringGLM.make_yX(f, nt_cat)
    @test y == [2.3, 3.4, 4.5, 5.4]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    # DataFrames
    f = @formula(y_int ~ 1 + x_float + x_cat)
    y, X = TuringGLM.make_yX(f, df_str)
    @test y == [2, 3, 4, 5]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    f = @formula(y_float ~ 1 + x_float + x_cat)
    y, X = TuringGLM.make_yX(f, df_str)
    @test y == [2.3, 3.4, 4.5, 5.4]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    f = @formula(y_int ~ 1 + x_float + x_cat_ordered)
    y, X = TuringGLM.make_yX(f, df_cat)
    @test y == [2, 3, 4, 5]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    f = @formula(y_float ~ 1 + x_float + x_cat_ordered)
    y, X = TuringGLM.make_yX(f, df_cat)
    @test y == [2.3, 3.4, 4.5, 5.4]
    @test X == [
        1.1 0.0 0.0 0.0
        2.3 1.0 0.0 0.0
        3.14 0.0 1.0 0.0
        3.65 0.0 0.0 1.0
    ]

    # center_predictors
    # NamedTuples
    f = @formula(y_int ~ 1 + x_float + x_cat)
    y, X = TuringGLM.make_yX(f, nt_str)
    μ_X, X_centered = TuringGLM.center_predictors(X)
    @test μ_X ≈ [2.5475 0.25 0.25 0.25]
    @test X_centered ≈ [
        -1.4475 -0.25 -0.25 -0.25
        -0.2475 0.75 -0.25 -0.25
        0.5925 -0.25 0.75 -0.25
        1.1025 -0.25 -0.25 0.75
    ]

    # DataFrames
    y, X = TuringGLM.make_yX(f, df_str)
    μ_X, X_centered = TuringGLM.center_predictors(X)
    @test μ_X ≈ [2.5475 0.25 0.25 0.25]
    @test X_centered ≈ [
        -1.4475 -0.25 -0.25 -0.25
        -0.2475 0.75 -0.25 -0.25
        0.5925 -0.25 0.75 -0.25
        1.1025 -0.25 -0.25 0.75
    ]

    # standardize_predictors
    # NamedTuples
    f = @formula(y_int ~ 1 + x_float + x_cat)
    y, X = TuringGLM.make_yX(f, nt_str)
    μ_X, σ_X, X_std = TuringGLM.standardize_predictors(X)
    @test μ_X ≈ [2.5475 0.25 0.25 0.25]
    @test σ_X ≈ [1.114013 0.5 0.5 0.5]
    @test X_std ≈ [
        -1.29935645 -0.5 -0.5 -0.5
        -0.222169756 1.5 -0.5 -0.5
        0.53186093 -0.5 1.5 -0.5
        0.98966527 -0.5 -0.5 1.5
    ]

    # DataFrames
    y, X = TuringGLM.make_yX(f, df_str)
    μ_X, X_centered = TuringGLM.center_predictors(X)
    μ_X, σ_X, X_std = TuringGLM.standardize_predictors(X)
    @test μ_X ≈ [2.5475 0.25 0.25 0.25]
    @test σ_X ≈ [1.114013 0.5 0.5 0.5]
    @test X_std ≈ [
        -1.29935645 -0.5 -0.5 -0.5
        -0.222169756 1.5 -0.5 -0.5
        0.53186093 -0.5 1.5 -0.5
        0.98966527 -0.5 -0.5 1.5
    ]
end
