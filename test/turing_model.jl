using StableRNGs: StableRNG

@timed_testset "turing_model" begin
    DATA_DIR = joinpath("..", "data")
    kidiq = CSV.read(joinpath(DATA_DIR, "kidiq.csv"), DataFrame)
    wells = CSV.read(joinpath(DATA_DIR, "wells.csv"), DataFrame)
    roaches = CSV.read(joinpath(DATA_DIR, "roaches.csv"), DataFrame)
    cheese = CSV.read(joinpath(DATA_DIR, "cheese.csv"), DataFrame)
    @timed_testset "Gaussian Model" begin
        f = @formula(kid_score ~ mom_iq * mom_hs)
        @testset "standardize=false" begin
            m = turing_model(f, kidiq)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 31.80 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.507 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.22 atol = 0.2
        end

        @testset "standardize=true" begin
            m = turing_model(f, kidiq; standardize=true)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 0.000 atol = 0.2
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.648 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.849 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(28, 5), nothing)
            m = turing_model(f, kidiq; priors)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 28.758 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.539 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.3863 atol = 0.2
        end
        @testset "explicit calling Normal" begin
            m = turing_model(f, kidiq; model=Normal)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 31.80 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.507 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.22 atol = 0.2
        end
    end
    @timed_testset "TDist Model" begin
        f = @formula(kid_score ~ mom_iq * mom_hs)
        @testset "standardize=false" begin
            m = turing_model(f, kidiq; model=TDist)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 33.31 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.519 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.340 atol = 0.2
            @test quantile(chn)[:ν, Symbol("50.0%")] ≈ 2.787 atol = 0.5
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(28, 5), Exponential(2))
            m = turing_model(f, kidiq; model=TDist, priors)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 28.565 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.551 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.255 atol = 0.2
            @test quantile(chn)[:ν, Symbol("50.0%")] ≈ 10.339 atol = 0.5
        end
    end
    @timed_testset "Bernoulli Model" begin
        f = @formula(switch ~ arsenic + dist + assoc + educ)
        @testset "standardize=false" begin
            m = turing_model(f, wells; model=Bernoulli)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ -0.153 atol = 0.2
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.467 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.009 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(), nothing)
            m = turing_model(f, wells; model=Bernoulli, priors)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ -0.155 atol = 0.2
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.468 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.009 atol = 0.2
        end
    end
    @timed_testset "Poisson Model" begin
        f = @formula(y ~ roach1 + treatment + senior + exposure2)
        @testset "standardize=false" begin
            m = turing_model(f, roaches; model=Poisson)
            # seed of 123 gives bad results
            chn = sample(StableRNG(124), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.969 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.006 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.5145 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(2, 5), Normal(), nothing)
            m = turing_model(f, roaches; model=Poisson, priors)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.963 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.006 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.5145 atol = 0.2
        end
    end
    @timed_testset "NegativeBinomial Model" begin
        f = @formula(y ~ roach1 + treatment + senior + exposure2)
        @testset "standardize=false" begin
            m = turing_model(f, roaches; model=NegativeBinomial)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.448 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.013 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.734 atol = 0.2
            @test quantile(chn)[:ϕ⁻, Symbol("50.0%")] ≈ 1.401 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(2, 5), Exponential(0.5))
            m = turing_model(f, roaches; model=NegativeBinomial, priors)
            chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.401 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.013 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.723 atol = 0.2
            @test quantile(chn)[:ϕ⁻, Symbol("50.0%")] ≈ 3.56 atol = 0.2
        end
    end
    @timed_testset "Hierarchical Model" begin
        f = @formula(y ~ (1 | cheese) + background)
        m = turing_model(f, cheese)
        chn = sample(StableRNG(123), m, NUTS(), MCMCThreads(), 2_000, 2)
        @test summarystats(chn)[:α, :mean] ≈ 68.07 atol = 2.0
        @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 6.60 atol = 0.2
        @test summarystats(chn)[Symbol("zⱼ[1]"), :mean] ≈ 0.348 atol = 0.2
        @test quantile(chn)[Symbol("zⱼ[2]"), Symbol("50.0%")] ≈ -1.376 atol = 0.5
    end
    @testset "Unsupported Model Likelihoods" begin
        @test_throws ArgumentError turing_model(@formula(y ~ x), nt_str; model=Binomial)
    end
    @testset "NegativeBinomial2" begin
        @test T.NegativeBinomial2(0, 1) == T.NegativeBinomial(1, 1)
        @test T.NegativeBinomial2(2, 8) == T.NegativeBinomial(8, 0.8)
        @test_throws DomainError T.NegativeBinomial2(0, -1)
    end
end
