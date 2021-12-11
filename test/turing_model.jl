@testset "turing_model.jl" begin
    kidiq = CSV.read(joinpath("..", "data", "kidiq.csv"), DataFrame)
    wells = CSV.read(joinpath("..", "data", "wells.csv"), DataFrame)
    roaches = CSV.read(joinpath("..", "data", "roaches.csv"), DataFrame)
    @testset "normal likelihood" begin
        @testset "standardize=false" begin
            normal_model = turing_model(@formula(kid_score ~ mom_iq * mom_hs), kidiq)
            normal_chn = sample(seed!(123), normal_model, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(normal_chn)[:α, :mean] ≈ 29.30 atol = 2.0
            @test summarystats(normal_chn)[Symbol("β[1]"), :mean] ≈ 0.533 atol = 0.2
            @test quantile(normal_chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.593 atol = 0.2
        end

        @testset "standardize=true" begin
            normal_model_std = turing_model(
                @formula(kid_score ~ mom_iq * mom_hs), kidiq; standardize=true
            )
            normal_chn_std = sample(
                seed!(123), normal_model_std, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(normal_chn_std)[:α, :mean] ≈ 0.000 atol = 0.2
            @test summarystats(normal_chn_std)[Symbol("β[1]"), :mean] ≈ 0.648 atol = 0.2
            @test quantile(normal_chn_std)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.849 atol =
                0.2
        end

        @testset "custom_priors" begin
            my_prior = CustomPrior(Normal(), Normal(28, 5), nothing)
            normal_model_prior = turing_model(
                @formula(kid_score ~ mom_iq * mom_hs), kidiq; priors=my_prior
            )
            normal_chn_prior = sample(
                seed!(123), normal_model_prior, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(normal_chn_prior)[:α, :mean] ≈ 28.758 atol = 2.0
            @test summarystats(normal_chn_prior)[Symbol("β[1]"), :mean] ≈ 0.539 atol = 0.2
            @test quantile(normal_chn_prior)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.3863 atol =
                0.2
        end
    end
    @testset "student likelihood" begin
        @testset "standardize=false" begin
            student_model = turing_model(
                @formula(kid_score ~ mom_iq * mom_hs), kidiq; family="student"
            )
            student_chn = sample(seed!(123), student_model, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(student_chn)[:α, :mean] ≈ 40.380 atol = 2.0
            @test summarystats(student_chn)[Symbol("β[1]"), :mean] ≈ 0.478 atol = 0.2
            @test quantile(student_chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.736 atol = 0.2
            @test quantile(student_chn)[:ν, Symbol("50.0%")] ≈ 1.039 atol = 0.5
        end

        @testset "custom_priors" begin
            my_prior = CustomPrior(Normal(), Normal(28, 5), Exponential(2))
            student_model_prior = turing_model(
                @formula(kid_score ~ mom_iq * mom_hs),
                kidiq;
                family="student",
                priors=my_prior,
            )
            student_chn_prior = sample(
                seed!(123), student_model_prior, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(student_chn_prior)[:α, :mean] ≈ 35.506 atol = 2.0
            @test summarystats(student_chn_prior)[Symbol("β[1]"), :mean] ≈ 0.522 atol = 0.2
            @test quantile(student_chn_prior)[Symbol("β[2]"), Symbol("50.0%")] ≈ 0.628 atol =
                0.2
            @test quantile(student_chn_prior)[:ν, Symbol("50.0%")] ≈ 1.178 atol = 0.5
        end
    end
    @testset "bernoulli likelihood" begin
        @testset "standardize=false" begin
            bernoulli_model = turing_model(
                @formula(switch ~ arsenic + dist + assoc + educ), wells; family="bernoulli"
            )
            bernoulli_chn = sample(
                seed!(123), bernoulli_model, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(bernoulli_chn)[:α, :mean] ≈ -0.153 atol = 0.2
            @test summarystats(bernoulli_chn)[Symbol("β[1]"), :mean] ≈ 0.467 atol = 0.2
            @test quantile(bernoulli_chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.009 atol =
                0.2
        end

        @testset "custom_priors" begin
            my_prior = CustomPrior(Normal(), Normal(), nothing)
            bernoulli_model_prior = turing_model(
                @formula(switch ~ arsenic + dist + assoc + educ),
                wells;
                family="bernoulli",
                priors=my_prior,
            )
            bernoulli_chn_prior = sample(
                seed!(123), bernoulli_model_prior, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(bernoulli_chn_prior)[:α, :mean] ≈ -0.155 atol = 0.2
            @test summarystats(bernoulli_chn_prior)[Symbol("β[1]"), :mean] ≈ 0.468 atol =
                0.2
            @test quantile(bernoulli_chn_prior)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.009 atol =
                0.2
        end
    end
    @testset "poisson likelihood" begin
        @testset "standardize=false" begin
            poisson_model = turing_model(
                @formula(y ~ roach1 + treatment + senior + exposure2),
                roaches;
                family="poisson",
            )
            poisson_chn = sample(seed!(123), poisson_model, NUTS(), MCMCThreads(), 2_000, 2)
            @test summarystats(poisson_chn)[:α, :mean] ≈ 2.969 atol = 0.5
            @test summarystats(poisson_chn)[Symbol("β[1]"), :mean] ≈ 0.006 atol = 0.2
            @test quantile(poisson_chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.5145 atol =
                0.2
        end

        @testset "custom_priors" begin
            my_prior = CustomPrior(Normal(2, 5), Normal(), nothing)
            poisson_model_prior = turing_model(
                @formula(y ~ roach1 + treatment + senior + exposure2),
                roaches;
                family="poisson",
                priors=my_prior,
            )
            poisson_chn_prior = sample(
                seed!(123), poisson_model_prior, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(poisson_chn_prior)[:α, :mean] ≈ 2.963 atol = 0.5
            @test summarystats(poisson_chn_prior)[Symbol("β[1]"), :mean] ≈ 0.006 atol = 0.2
            @test quantile(poisson_chn_prior)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.5145 atol =
                0.2
        end
    end
    @testset "negative binomial likelihood" begin
        @testset "standardize=false" begin
            negativebinomial_model = turing_model(
                @formula(y ~ roach1 + treatment + senior + exposure2),
                roaches;
                family="negativebinomial",
            )
            negativebinomial_chn = sample(
                seed!(123), negativebinomial_model, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(negativebinomial_chn)[:α, :mean] ≈ 2.448 atol = 0.5
            @test summarystats(negativebinomial_chn)[Symbol("β[1]"), :mean] ≈ 0.013 atol =
                0.2
            @test quantile(negativebinomial_chn)[Symbol("β[2]"), Symbol("50.0%")] ≈ -0.734 atol =
                0.2
            @test quantile(negativebinomial_chn)[:ϕ⁻, Symbol("50.0%")] ≈ 1.401 atol = 0.2
        end

        @testset "custom_priors" begin
            my_prior = CustomPrior(Normal(), Normal(2, 5), Exponential(0.5))
            negativebinomial_model_prior = turing_model(
                @formula(y ~ roach1 + treatment + senior + exposure2),
                roaches;
                family="negativebinomial",
                priors=my_prior,
            )
            negativebinomial_chn_prior = sample(
                seed!(123), negativebinomial_model_prior, NUTS(), MCMCThreads(), 2_000, 2
            )
            @test summarystats(negativebinomial_chn_prior)[:α, :mean] ≈ 2.422 atol = 0.5
            @test summarystats(negativebinomial_chn_prior)[Symbol("β[1]"), :mean] ≈ 0.013 atol =
                0.2
            @test quantile(negativebinomial_chn_prior)[Symbol("β[2]"), Symbol("50.0%")] ≈
                -0.732 atol = 0.2
            @test quantile(negativebinomial_chn_prior)[:ϕ⁻, Symbol("50.0%")] ≈ 3.56 atol =
                0.2
        end
    end
    @testset "NegativeBinomial2" begin
        @test T.NegativeBinomial2(0, 1) == T.NegativeBinomial(1, 1)
        @test T.NegativeBinomial2(2, 8) == T.NegativeBinomial(8, 0.8)
        @test_throws ArgumentError T.NegativeBinomial2(0, -1)
    end
end
