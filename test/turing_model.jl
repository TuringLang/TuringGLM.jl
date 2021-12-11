@testset "turing_model.jl" begin
    kidiq = CSV.read(joinpath("..", "data", "kidiq.csv"), DataFrame)
    wells = CSV.read(joinpath("..", "data", "wells.csv"), DataFrame)
    roaches = CSV.read(joinpath("..", "data", "roaches.csv"), DataFrame)
    @testset "normal likelihood" begin
        @testset "standardize=false" begin
            normal_model = turing_model(@formula(kid_score ~ mom_iq * mom_hs), kidiq)
            normal_chn = sample(normal_model, NUTS(), MCMCThreads(), 2_000, 2)
            @show normal_chn
        end

        @testset "standardize=true" begin
        end
    end
    @testset "student likelihood" begin end
    @testset "bernoulli likelihood" begin end
    @testset "poisson likelihood" begin end
    @testset "negative binomial likelihood" begin end
    @testset "NegativeBinomial2" begin
        @test T.NegativeBinomial2(0, 1) == T.NegativeBinomial(1, 1)
        @test T.NegativeBinomial2(2, 8) == T.NegativeBinomial(8, 0.8)
        @test_throws ArgumentError T.NegativeBinomial2(-10, 1)
        @test_throws ArgumentError T.NegativeBinomial2(0, -1)
        @test_throws ArgumentError T.NegativeBinomial2(-10, -1)
    end
end
