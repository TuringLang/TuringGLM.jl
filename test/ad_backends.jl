@timed_testset "ad_backends.jl" begin
    DATA_DIR = joinpath("..", "data")
    cheese = CSV.read(joinpath(DATA_DIR, "cheese.csv"), DataFrame)
    f = @formula(y ~ (1 | cheese) + background)
    m = turing_model(f, cheese)

    ADTYPES = [
        AutoForwardDiff(),
        AutoReverseDiff(; compile=false),
        AutoReverseDiff(; compile=true),
        AutoMooncake(; config=nothing),
    ]
    @testset "$adtype" for adtype in ADTYPES
        @test sample(m, NUTS(; adtype=adtype), 20) isa Chains
    end
end
