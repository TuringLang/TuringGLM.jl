@testset "utils" begin
    @testset "center_predictors" begin
        @testset "NamedTuples" begin
            f = @formula(y_int ~ x_float + x_cat)
            X = T.data_fixed_effects(f, nt_str)
            μ_X, X_centered = T.center_predictors(X)
            @test μ_X ≈ [2.547 0.25 0.25 0.25] atol = 0.01
            @test X_centered ≈ [
                -1.447 -0.25 -0.25 -0.25
                -0.247 0.75 -0.25 -0.25
                0.592 -0.25 0.75 -0.25
                1.102 -0.25 -0.25 0.75
            ] atol = 0.01
            @test mapslices(mean, X_centered; dims=1) ≈ [0 0 0 0] atol = 0.01
        end

        @testset "DataFrames" begin
            f = @formula(y_int ~ x_float + x_cat)
            X = T.data_fixed_effects(f, df_str)
            μ_X, X_centered = T.center_predictors(X)
            @test μ_X ≈ [2.547 0.25 0.25 0.25] atol = 0.01
            @test X_centered ≈ [
                -1.447 -0.25 -0.25 -0.25
                -0.247 0.75 -0.25 -0.25
                0.592 -0.25 0.75 -0.25
                1.102 -0.25 -0.25 0.75
            ] atol = 0.01
            @test mapslices(mean, X_centered; dims=1) ≈ [0 0 0 0] atol = 0.01
        end
    end

    @testset "standardize_predictors" begin
        @testset "NamedTuples" begin
            f = @formula(y_int ~ x_float + x_cat)
            X = T.data_fixed_effects(f, nt_str)
            μ_X, σ_X, X_std = T.standardize_predictors(X)
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

        @testset "DataFrames" begin
            f = @formula(y_int ~ x_float + x_cat)
            X = T.data_fixed_effects(f, df_str)
            μ_X, σ_X, X_std = T.standardize_predictors(X)
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
end
