@timed_testset "ad_backends" begin
    DATA_DIR = joinpath("..", "data")
    cheese = CSV.read(joinpath(DATA_DIR, "cheese.csv"), DataFrame)
    f = @formula(y ~ (1 | cheese) + background)
    m = turing_model(f, cheese)
    # only running 2 samples to test if the different ADs runs
    @timed_testset "ForwardDiff" begin
        chn = sample(m, NUTS(; adtype=AutoForwardDiff(; chunksize=8)), 2)
        @test chn isa Chains
    end
    # TODO: fix Tracker tests
    # @timed_testset "Tracker" begin
    #     using Tracker
    #     chn = sample(m, NUTS(; adtype=AutoTracker()), 2)
    #     @test chn isa Chains
    # end
    # TODO: fix Zygote tests
    # @timed_testset "Zygote" begin
    #     using Zygote
    #     chn = sample(m, NUTS(; adtype=AutoZygote()), 2)
    #     @test chn isa Chains
    # end
    @timed_testset "ReverseDiff" begin
        using ReverseDiff
        chn = sample(m, NUTS(; adtype=AutoReverseDiff(; compile=true)), 2)
        @test chn isa Chains
    end
end
