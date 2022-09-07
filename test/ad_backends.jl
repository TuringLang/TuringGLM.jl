@timed_testset "ad_backends" begin
    DATA_DIR = joinpath("..", "data")
    cheese = CSV.read(joinpath(DATA_DIR, "cheese.csv"), DataFrame)
    f = @formula(y ~ (1 | cheese) + background)
    m = turing_model(f, cheese)
    # only running 2 samples to test if the different ADs runs
    @timed_testset "ForwardDiff" begin
        Turing.setadbackend(:forwarddiff)
        chn = sample(m, NUTS(), 2)
        @test chn isa Chains
    end
    # TODO
    # FIXME
    # @timed_testset "Tracker" begin
    #     using Tracker
    #     Turing.setadbackend(:tracker)
    #     chn = sample(m, NUTS(), 2)
    #     @test chn isa Chains
    # end
    @timed_testset "Zygote" begin
        using Zygote
        Turing.setadbackend(:zygote)
        chn = sample(m, NUTS(), 2)
        @test chn isa Chains
    end
    @timed_testset "ReverseDiff" begin
        using ReverseDiff
        Turing.setadbackend(:reversediff)
        chn = sample(m, NUTS(), 2)
        @test chn isa Chains
    end
    # go back to defaults
    Turing.setadbackend(:forwarddiff)
end
