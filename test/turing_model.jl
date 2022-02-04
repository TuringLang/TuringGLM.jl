@testset "turing_model.jl" begin
    DATA_DIR = joinpath("..", "data")
    kidiq = CSV.read(joinpath(DATA_DIR, "kidiq.csv"), DataFrame)
    wells = CSV.read(joinpath(DATA_DIR, "wells.csv"), DataFrame)
    roaches = CSV.read(joinpath(DATA_DIR, "roaches.csv"), DataFrame)
    cheese = CSV.read(joinpath(DATA_DIR, "cheese.csv"), DataFrame)
    @testset "Gaussian Model" begin
        f = @formula(kid_score ~ mom_iq * mom_hs)
        @testset "standardize=false" begin
            m = turing_model(f, kidiq)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 29.30 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.533 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.593 atol = 0.2
        end

        @testset "standardize=true" begin
            m = turing_model(f, kidiq; standardize=true)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 0.000 atol = 0.2
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.648 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.849 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(28, 5), nothing)
            m = turing_model(f, kidiq; priors)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 28.758 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.539 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.3863 atol = 0.2
        end
        @testset "explicit calling Gaussian" begin
            m = turing_model(f, kidiq, Gaussian())
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 29.30 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.533 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.593 atol = 0.2
        end
    end
    @testset "Student Model" begin
        f = @formula(kid_score ~ mom_iq * mom_hs)
        @testset "standardize=false" begin
            m = turing_model(f, kidiq, Student())
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 40.380 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.478 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.736 atol = 0.2
            @test quantile(chn)[:ν, Symbol("50.0%")] ≈ 1.039 atol = 0.5
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(28, 5), Exponential(2))
            m = turing_model(f, kidiq, Student(); priors)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 35.506 atol = 2.0
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.522 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.628 atol = 0.2
            @test quantile(chn)[:ν, Symbol("50.0%")] ≈ 1.178 atol = 0.5
        end
    end
    @testset "Logistic Model" begin
        f = @formula(switch ~ arsenic + dist + assoc + educ)
        @testset "standardize=false" begin
            m = turing_model(f, wells, Logistic())
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ -0.153 atol = 0.2
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.467 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.009 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(), nothing)
            m = turing_model(f, wells, Logistic(); priors)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ -0.155 atol = 0.2
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.468 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.009 atol = 0.2
        end
    end
    @testset "Pois Model" begin
        f = @formula(y ~ roach1 + treatment + senior + exposure2)
        @testset "standardize=false" begin
            m = turing_model(f, roaches, Pois())
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.969 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.006 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.5145 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(2, 5), Normal(), nothing)
            m = turing_model(f, roaches, Pois(); priors)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.963 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.006 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.5145 atol = 0.2
        end
    end
    @testset "NegBin Model" begin
        f = @formula(y ~ roach1 + treatment + senior + exposure2)
        @testset "standardize=false" begin
            m = turing_model(f, roaches, NegBin())
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.448 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.013 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.734 atol = 0.2
            @test quantile(chn)[:ϕ⁻, Symbol("50.0%")] ≈ 1.401 atol = 0.2
        end

        @testset "custom_priors" begin
            priors = CustomPrior(Normal(), Normal(2, 5), Exponential(0.5))
            m = turing_model(f, roaches, NegBin(); priors)
            chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(chn)[:α, :mean] ≈ 2.422 atol = 0.5
            @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 0.013 atol = 0.2
            @test quantile(chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.732 atol = 0.2
            @test quantile(chn)[:ϕ⁻, Symbol("50.0%")] ≈ 3.56 atol = 0.2
        end
    end
    @testset "Hierarchical Model" begin
        f = @formula(y ~ (1 | cheese) + background)
        m = turing_model(f, cheese)
        chn = sample(seed!(123), m, NUTS(), MCMCThreads(), 2_000, 2)
        @test summarystats(chn)[:α, :mean] ≈ 68.33 atol = 2.0
        @test summarystats(chn)[Symbol("β[1]"), :mean] ≈ 6.928 atol = 0.2
        @test summarystats(chn)[Symbol("zⱼ[1]"), :mean] ≈ 0.306 atol = 0.2
        @test quantile(chn)[Symbol("zⱼ[2]"), Symbol("50.0%")] ≈ -1.422 atol = 0.5
    end
    @testset "Unsupported Model Likelihoods" begin
        @test_throws ArgumentError turing_model(@formula(y ~ x), nt_str, Normal())
        @test_throws ArgumentError turing_model(@formula(y ~ x), nt_str, Binomial())
    end
    @testset "NegativeBinomial2" begin
        @test T.NegativeBinomial2(0, 1) == T.NegativeBinomial(1, 1)
        @test T.NegativeBinomial2(2, 8) == T.NegativeBinomial(8, 0.8)
        @test_throws DomainError T.NegativeBinomial2(0, -1)
    end
end
