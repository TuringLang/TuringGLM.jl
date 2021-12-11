@testset "turing_model.jl" begin
    @testset "normal likelihood" begin
    end
    @testset "normal likelihood" begin
    end
    @testset "student likelihood" begin
    end
    @testset "bernoulli likelihood" begin
    end
    @testset "poisson likelihood" begin
    end
    @testset "negative binomial likelihood" begin
    end
    @testset "NegativeBinomial2" begin
        @test T.NegativeBinomial2(0, 1) == T.NegativeBinomial(1, 1)
        @test T.NegativeBinomial2(2, 8) == T.NegativeBinomial(8, 0.8)
        @test_throws ArgumentError T.NegativeBinomial2(-10, 1)
        @test_throws ArgumentError T.NegativeBinomial2(0, -1)
        @test_throws ArgumentError T.NegativeBinomial2(-10, -1)
    end
end
