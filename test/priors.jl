@testset "prior.jl" begin

    @testset "types" begin
        @test DefaultPrior() isa T.Prior

        my_prior = CustomPrior(Normal(), Normal(), nothing)
        @test my_prior isa T.Prior

        my_prior = CustomPrior(Normal(), Normal(), Exponential(1))
        @test my_prior isa T.Prior
    end

    @testset "constructors" begin
        @test_throws ArgumentError DefaultPrior(Normal(0, 1))
        @test_throws ArgumentError CustomPrior()
    end

    @testset "CustomPrior" begin
        my_prior = CustomPrior(Normal(), Normal(), nothing)
        @test my_prior.predictors == Normal(0, 1)
        @test my_prior.intercept == Normal(0, 1)
        @test isnothing(my_prior.auxiliary)

        my_prior = CustomPrior(Normal(), Normal(), Exponential(1))
        @test my_prior.predictors == Normal(0, 1)
        @test my_prior.intercept == Normal(0, 1)
        @test my_prior.auxiliary == Exponential(1)
    end
end
