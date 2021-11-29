function mimestring(mime::Type{<:MIME}, x)
    buf = IOBuffer()
    show(buf, mime(), x)
    return String(take!(buf))
end
mimestring(x) = mimestring(MIME"text/plain", x)

@testset "terms" begin
    @testset "concrete_term" begin
        t = T.term(:aaa)
        ts = T.term("aaa")
        @test t == ts
        @test string(t) == "aaa"
        @test mimestring(t) == "aaa(unknown)"

        t0 = T.concrete_term(t, [3, 2, 1])
        @test string(t0) == "aaa"
        @test mimestring(t0) == "aaa(continuous)"
        @test t0.mean == 2.0
        @test t0.var == var([1, 2, 3])
        @test t0.min == 1.0
        @test t0.max == 3.0

        t1 = T.concrete_term(t, [:a, :b, :c])
        @test t1.contrasts isa T.ContrastsMatrix{T.DummyCoding}
        @test string(t1) == "aaa"
        @test mimestring(t1) == "aaa(TuringGLM.DummyCoding:3→2)"

        t3 = T.concrete_term(t, [:a, :b, :c], T.DummyCoding())
        @test t3.contrasts isa T.ContrastsMatrix{T.DummyCoding}
        @test string(t3) == "aaa"
        @test mimestring(t3) == "aaa(TuringGLM.DummyCoding:3→2)"

        t2full = T.concrete_term(t, [:a, :a, :b], T.FullDummyCoding())
        @test t2full.contrasts isa T.ContrastsMatrix{T.FullDummyCoding}
        @test mimestring(t2full) == "aaa(TuringGLM.FullDummyCoding:2→2)"
        @test string(t2full) == "aaa"
    end

    @testset "term operators" begin
        a = T.term(:a)
        b = T.term(:b)
        @test a + b == (a, b)
        @test (a ~ b) == T.FormulaTerm(a, b)
        @test string(a ~ b) == "$a ~ $b"
        @test mimestring(a ~ b) == """FormulaTerm
                                      Response:
                                        a(unknown)
                                      Predictors:
                                        b(unknown)"""
        @test mimestring(a ~ T.term(1) + b) == """FormulaTerm
                                                Response:
                                                  a(unknown)
                                                Predictors:
                                                  1
                                                  b(unknown)"""
        @test a & b == T.InteractionTerm((a, b))
        @test string(a & b) == "$a & $b"
        @test mimestring(a & b) == "a(unknown) & b(unknown)"
        c = T.term(:c)
        ab = a + b
        bc = b + c
        abc = a + b + c
        @test ab + c == abc
        @test ab + a == ab
        @test a + bc == abc
        @test b + ab == ab
        @test ab + ab == ab
        @test ab + bc == abc
        @test sum((a, b, c)) == abc
        @test sum((a,)) == a
        @test +a == a
    end

    @testset "expand nested tuples of terms during apply_schema" begin
        sch = T.schema((a=rand(10), b=rand(10), c=rand(10)))

        # nested tuples of terms are expanded by T.apply_schema
        terms = (T.term(:a), (T.term(:b), T.term(:c)))
        terms2 = T.apply_schema(terms, sch)
        @test terms2 isa NTuple{3,T.ContinuousTerm}
        @test terms2 == T.apply_schema(T.term.((:a, :b, :c)), sch)
    end

    @testset "Intercept and response traits" begin
        has_responses = [
            T.term(:y),
            T.term(1),
            T.InterceptTerm{true}(),
            T.term(:y) + T.term(:z),
            T.term(:y) + T.term(0),
            T.term(:y) + T.InterceptTerm{false}(),
        ]
        no_responses = [T.term(0), T.InterceptTerm{false}()]

        has_intercepts = [T.term(1), T.InterceptTerm{true}()]
        omits_intercepts = [T.term(0), T.term(-1), T.InterceptTerm{false}()]

        a = T.term(:a)

        for lhs in has_responses, rhs in has_intercepts
            @test T.hasresponse(lhs ~ rhs)
            @test T.hasintercept(lhs ~ rhs)
            @test !T.omitsintercept(lhs ~ rhs)

            @test T.hasresponse(lhs ~ rhs + a)
            @test T.hasintercept(lhs ~ rhs + a)
            @test !T.omitsintercept(lhs ~ rhs + a)
        end

        for lhs in no_responses, rhs in has_intercepts
            @test !T.hasresponse(lhs ~ rhs)
            @test T.hasintercept(lhs ~ rhs)
            @test !T.omitsintercept(lhs ~ rhs)

            @test !T.hasresponse(lhs ~ rhs + a)
            @test T.hasintercept(lhs ~ rhs + a)
            @test !T.omitsintercept(lhs ~ rhs + a)
        end

        for lhs in has_responses, rhs in omits_intercepts
            @test T.hasresponse(lhs ~ rhs)
            @test !T.hasintercept(lhs ~ rhs)
            @test T.omitsintercept(lhs ~ rhs)

            @test T.hasresponse(lhs ~ rhs + a)
            @test !T.hasintercept(lhs ~ rhs + a)
            @test T.omitsintercept(lhs ~ rhs + a)
        end

        for lhs in no_responses, rhs in omits_intercepts
            @test !T.hasresponse(lhs ~ rhs)
            @test !T.hasintercept(lhs ~ rhs)
            @test T.omitsintercept(lhs ~ rhs)

            @test !T.hasresponse(lhs ~ rhs + a)
            @test !T.hasintercept(lhs ~ rhs + a)
            @test T.omitsintercept(lhs ~ rhs + a)
        end
    end

    @testset "Tuple terms" begin
        a, b, c = T.Term.((:a, :b, :c))

        # TermOrTerms - one or more AbstractTerms (if more, a tuple)
        # empty tuples are never terms
        @test !(() isa T.TermOrTerms)
        @test (a,) isa T.TermOrTerms
        @test (a, b) isa T.TermOrTerms
        @test (a, b, a & b) isa T.TermOrTerms
        @test !(((), a) isa T.TermOrTerms)
        # can't contain further tuples
        @test !((a, (a,), b) isa T.TermOrTerms)

        # a tuple of AbstractTerms OR Tuples of one or more terms
        # empty tuples are never terms
        @test !(() isa T.TupleTerm)
        @test (a,) isa T.TupleTerm
        @test (a, b) isa T.TupleTerm
        @test (a, b, a & b) isa T.TupleTerm
        @test !(((), a) isa T.TupleTerm)
        @test (((a,), a) isa T.TupleTerm)

        # no methods for operators on term and empty tuple (=no type piracy)
        @test_throws MethodError a + ()
        @test_throws MethodError () + a
        @test_throws MethodError a & ()
        @test_throws MethodError () & a
        @test_throws MethodError a ~ ()
        @test_throws MethodError () ~ a

        # show methods of empty tuples preserved
        @test "$(())" == "()"
        @test "$((a,b))" == "a + b"
        @test "$((a, ()))" == "(a, ())"
    end

    @testset "concrete_term error messages" begin
        t = (a=[1, 2, 3], b=[0.0, 0.5, 1.0])
        @test Tables.istable(t)
        @test_throws ArgumentError T.concrete_term(T.term(:not_there), t)
    end

    @testset "coefnames" begin
        f = @formula y_float ~ 0 + x_int + x_cat
        sch = T.schema(f, nt_str)
        ts = T.apply_schema(f.rhs, sch)
        ts = T.collect_matrix_terms(ts)
        coef_names = T.coefnames(ts)
        @test coef_names == ["x_int", "x_cat: 2", "x_cat: 3", "x_cat: 4"]

        f = @formula y_float ~ 1 + x_int + x_cat
        sch = T.schema(f, nt_str)
        ts = T.apply_schema(f.rhs, sch)
        ts = T.collect_matrix_terms(ts)
        coef_names = T.coefnames(ts)
        @test coef_names == ["x_int", "x_cat: 2", "x_cat: 3", "x_cat: 4"]

        f = @formula y_float ~ 0 + x_float * x_cat_ordered
        sch = T.schema(f, nt_cat)
        ts = T.apply_schema(f.rhs, sch)
        ts = T.collect_matrix_terms(ts)
        coef_names = T.coefnames(ts)
        @test coef_names == [
            "x_float",
            "x_cat_ordered: 2",
            "x_cat_ordered: 3",
            "x_cat_ordered: 4",
            "x_float & x_cat_ordered: 2",
            "x_float & x_cat_ordered: 3",
            "x_float & x_cat_ordered: 4",
        ]

        f = @formula y_float ~ 1 + x_float * x_cat_ordered
        sch = T.schema(f, nt_cat)
        ts = T.apply_schema(f.rhs, sch)
        ts = T.collect_matrix_terms(ts)
        coef_names = T.coefnames(ts)
        @test coef_names == [
            "x_float",
            "x_cat_ordered: 2",
            "x_cat_ordered: 3",
            "x_cat_ordered: 4",
            "x_float & x_cat_ordered: 2",
            "x_float & x_cat_ordered: 3",
            "x_float & x_cat_ordered: 4",
        ]

        # Interactions coming first
        f = @formula y_float ~ 1 + x_int * x_cat + x_float
        sch = T.schema(f, nt_str)
        ts = T.apply_schema(f.rhs, sch)
        ts = T.collect_matrix_terms(ts)
        coef_names = T.coefnames(ts)
        @test coef_names == [
            "x_int",
            "x_cat: 2",
            "x_cat: 3",
            "x_cat: 4",
            "x_float",
            "x_int & x_cat: 2",
            "x_int & x_cat: 3",
            "x_int & x_cat: 4",
        ]

        f = @formula y_float ~ 1 + x_int * x_cat_ordered + x_float
        sch = T.schema(f, nt_cat)
        ts = T.apply_schema(f.rhs, sch)
        ts = T.collect_matrix_terms(ts)
        coef_names = T.coefnames(ts)
        @test coef_names == [
            "x_int",
            "x_cat_ordered: 2",
            "x_cat_ordered: 3",
            "x_cat_ordered: 4",
            "x_float",
            "x_int & x_cat_ordered: 2",
            "x_int & x_cat_ordered: 3",
            "x_int & x_cat_ordered: 4",
        ]
    end
end
