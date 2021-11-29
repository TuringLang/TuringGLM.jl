@testset "formula" begin
    y, x1, x2, x3, a, b, c, onet = T.term.((:y, :x1, :x2, :x3, :a, :b, :c, 1))

    @testset "terms" begin

        # totally empty
        f = @formula(0 ~ 0)
        @test !T.hasresponse(f)
        @test !T.hasintercept(f)
        @test T.omitsintercept(f)
        @test f.rhs == T.ConstantTerm(0)
        @test issetequal(T.terms(f), [T.ConstantTerm(0)])

        # empty lhs, intercept on rhs
        f = @formula(0 ~ 1)
        @test !T.hasresponse(f)
        @test T.hasintercept(f)
        @test !T.omitsintercept(f)

        # empty RHS
        f = @formula(y ~ 0)
        @test T.hasintercept(f) == false
        @test T.omitsintercept(f) == true
        @test T.hasresponse(f)
        @test f.rhs == T.ConstantTerm(0)
        @test issetequal(T.terms(f), T.term.((:y, 0)))

        f = @formula(y ~ -1)
        @test T.hasintercept(f) == false
        @test T.omitsintercept(f) == true

        # intercept-only
        f = @formula(y ~ 1)
        @test T.hasresponse(f) == true
        @test T.hasintercept(f) == true
        @test f.rhs == onet
        @test issetequal(T.terms(f), (onet, y))

        # simple formula
        f = @formula y ~ 1 + x1
        @test T.hasresponse(f) == true
        @test T.hasintercept(f) == true

        # terms add
        f = @formula y ~ 1 + x1 + x2
        @test T.hasresponse(f) == true
        @test f.rhs == (onet, x1, x2)
        @test issetequal(T.terms(f), [y, onet, x1, x2])

        # implicit intercept behavior: NO intercept after @formula
        f = @formula(y ~ x1 + x2)
        @test T.hasintercept(f) == false
        @test T.omitsintercept(f) == false
        @test f.rhs == (x1, x2)
        @test issetequal(T.terms(f), [y, x1, x2])

        # no intercept
        f = @formula(y ~ 0 + x1 + x2)
        @test T.hasintercept(f) == false
        @test T.omitsintercept(f) == true
        @test f.rhs == T.term.((0, :x1, :x2))

        f = @formula(y ~ -1 + x1 + x2)
        @test T.hasintercept(f) == false
        @test T.omitsintercept(f) == true
        @test f.rhs == T.term.((-1, :x1, :x2))

        f = @formula(y ~ x1 & x2)
        @test T.hasintercept(f) == false
        @test T.omitsintercept(f) == false
        @test f.rhs == x1 & x2
        @test issetequal(T.terms(f), [y, x1, x2])

        # plain interaction
        f = @formula y ~ x1 & x2
        @test f.rhs == x1 & x2
        @test issetequal(T.terms(f), [y, x1, x2])

        # `*` main effects and interactions expansion
        f = @formula(y ~ x1 * x2)
        @test f.rhs == (x1, x2, x1 & x2)
        @test issetequal(T.terms(f), [y, x1, x2])

        # Incorrect formula separator
        @test_throws LoadError @eval @formula(y => x1 + x2)
    end

    @testset "associative/distributive rule" begin
        # `+`
        f = @formula(y ~ x1 + x2 + x3)
        @test f.rhs == (x1, x2, x3)

        # `&`
        f = @formula(y ~ x1 & x2 & x3)
        @test f.rhs == x1 & x2 & x3
        @test issetequal(T.terms(f), [y, x1, x2, x3])

        # distributive property of `+` and `&`
        f = @formula(y ~ x1 & (x2 + x3))
        @test f.rhs == (x1 & x2, x1 & x3)
        @test issetequal(T.terms(f), [y, x1, x2, x3])

        # ordering of interaction terms is preserved across distributive
        f = @formula(y ~ (x2 + x3) & x1)
        @test f.rhs == x2 & x1 + x3 & x1

        # distributive with `*`
        f = @formula(y ~ (a + b) * c)
        @test f.rhs == (a, b, c, a & c, b & c)

        # three-way `*`
        f = @formula(y ~ a * b * c)
        @test f.rhs == (a, b, c, a & b, a & c, b & c, a & b & c)
        @test issetequal(T.terms(f), (y, a, b, c))

        # Interactions with `1` reduce to main effect.
        f = @formula(y ~ 1 & x1)
        @test f.rhs == x1

        f = @formula(y ~ (1 + x1) & x2)
        @test f.rhs == (x2, x1 & x2)
    end

    @testset "drop_terms" begin
        form = @formula(foo ~ 1 + bar + baz)
        @test form == @formula(foo ~ 1 + bar + baz)
        T.drop_term(form, T.term(:bar))
        # drop_term creates a new formula:
        @test form != @formula(foo ~ 1 + baz)
    end

    @testset "copying formulas" begin
        f = @formula(foo ~ baz)
        @test f == deepcopy(f)

        f = @formula(foo ~ 1 + bar)
        @test f == deepcopy(f)

        f = @formula(foo ~ bar + baz)
        @test f == deepcopy(f)
    end
end
