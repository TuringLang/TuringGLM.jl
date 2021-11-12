@testset "formula" begin
    using TuringGLM: has_response, term, terms, drop_term

    y, x1, x2, x3, a, b, c = term.((:y, :x1, :x2, :x3, :a, :b, :c))

    @testset "terms" begin

        # we do not support intercepts
        @test_throws MethodError term(0)
        @test_throws MethodError term(1)
        @test_throws MethodError term(-1)

        # simple formula
        f = @formula y ~ x1
        @test has_response(f) == true

        # terms add
        f = @formula y ~ x1 + x2
        @test has_response(f) == true
        @test f.rhs == (x1, x2)
        @test issetequal(terms(f), [y, x1, x2])

        # plain interaction
        f = @formula y ~ x1 & x2
        @test f.rhs == x1 & x2
        @test issetequal(terms(f), [y, x1, x2])

        # `*` main effects and interactions expansion
        f = @formula(y ~ x1 * x2)
        @test f.rhs == (x1, x2, x1 & x2)
        @test issetequal(terms(f), [y, x1, x2])

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
        @test issetequal(terms(f), [y, x1, x2, x3])

        # distributive property of `+` and `&`
        f = @formula(y ~ x1 & (x2 + x3))
        @test f.rhs == (x1 & x2, x1 & x3)
        @test issetequal(terms(f), [y, x1, x2, x3])

        # ordering of interaction terms is preserved across distributive
        f = @formula(y ~ (x2 + x3) & x1)
        @test f.rhs == x2 & x1 + x3 & x1

        # distributive with `*`
        f = @formula(y ~ (a + b) * c)
        @test f.rhs == (a, b, c, a & c, b & c)

        # three-way `*`
        f = @formula(y ~ a * b * c)
        @test f.rhs == (a, b, c, a & b, a & c, b & c, a & b & c)
        @test issetequal(terms(f), (y, a, b, c))
    end

    @testset "drop_terms" begin
        form = @formula(foo ~ bar + baz)
        @test form == @formula(foo ~ bar + baz)
        drop_term(form, term(:bar))
        # drop_term creates a new formula:
        @test form != @formula(foo ~ baz)
    end

    @testset "copying formulas" begin
        f = @formula(foo ~ bar)
        @test f == deepcopy(f)

        f = @formula(foo ~ bar + baz)
        @test f == deepcopy(f)
    end
end
