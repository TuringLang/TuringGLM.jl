@testset "schema" begin
    @testset "no-op apply_schema" begin
        f = @formula(y ~ 1 + a + b + c + b & c)
        df = (y=rand(9), a=1:9, b=rand(9), c=repeat(["d", "e", "f"], 3))
        f = T.apply_schema(f, T.schema(f, df))
        @test f == T.apply_schema(f, T.schema(f, df))
    end

    @testset "lonely term in a tuple" begin
        d = (a=[1, 1],)
        @test T.apply_schema(T.ConstantTerm(1), T.schema(d)) ==
            T.apply_schema((T.ConstantTerm(1),), T.schema(d))
        @test T.apply_schema(T.Term(:a), T.schema(d)) ==
            T.apply_schema((T.Term(:a),), T.schema(d))
    end

    @testset "hints" begin
        f = @formula(y ~ 1 + a)
        d = (y=rand(10), a=repeat([1, 2]; outer=2))

        sch = T.schema(f, d)
        @test sch[T.term(:a)] isa T.ContinuousTerm

        sch1 = T.schema(f, d, Dict(:a => T.CategoricalTerm))
        @test sch1[T.term(:a)] isa T.CategoricalTerm{T.DummyCoding}
        f1 = T.apply_schema(f, sch1)
        @test f1.rhs.terms[end] == sch1[T.term(:a)]

        sch2 = T.schema(f, d, Dict(:a => T.DummyCoding()))
        @test sch2[T.term(:a)] isa T.CategoricalTerm{T.DummyCoding}
        f2 = T.apply_schema(f, sch2)
        @test f2.rhs.terms[end] == sch2[T.term(:a)]

        hint = deepcopy(sch2[T.term(:a)])
        sch3 = T.schema(f, d, Dict(:a => hint))
        # if an <:AbstractTerm is supplied as hint, it's included as is
        @test sch3[T.term(:a)] === hint !== sch2[T.term(:a)]
        f3 = T.apply_schema(f, sch3)
        @test f3.rhs.terms[end] === hint
    end

    @testset "has_schema" begin
        d = (y=rand(10), a=rand(10), b=repeat([:a, :b], 5))

        f = @formula(y ~ a * b)
        @test !T.has_schema(f)
        @test !T.has_schema(f.rhs)
        @test !T.has_schema(T.collect_matrix_terms(f.rhs))

        ff = T.apply_schema(f, T.schema(d))
        @test T.has_schema(ff)
        @test T.has_schema(ff.rhs)
        @test T.has_schema(T.collect_matrix_terms(ff.rhs))

        sch = T.schema(d)
        a, b = T.term.((:a, :b))
        @test !T.has_schema(a)
        @test T.has_schema(sch[a])
        @test !T.has_schema(b)
        @test T.has_schema(sch[b])

        @test !T.has_schema(a & b)
        @test !T.has_schema(a & sch[b])
        @test !T.has_schema(sch[a] & a)
        @test T.has_schema(sch[a] & sch[b])
    end
end
