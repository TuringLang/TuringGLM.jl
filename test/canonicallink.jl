@testset "canonicallink.jl" begin
    @test T.canonicallink(Bernoulli()) == T.LogitLink()
    @test T.canonicallink(Binomial()) == T.LogitLink()
    @test T.canonicallink(Gamma()) == T.InverseLink()
    @test T.canonicallink(InverseGaussian()) == T.InverseSquareLink()
    @test T.canonicallink(NegativeBinomial()) == T.NegativeBinomialLink(1)
    @test T.canonicallink(Normal()) == T.IdentityLink()
    @test T.canonicallink(Poisson()) == T.LogLink()
end
