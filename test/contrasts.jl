@testset "contrats" begin
    @testset "DummyCoding" begin
        levs = [:a, :b, :c, :d]
        base = [:c]

        @test T.baselevel(T.DummyCoding()) == nothing
        @test levels(T.DummyCoding()) == nothing

        @test T.baselevel(T.DummyCoding(; levels=levs)) == nothing
        @test levels(T.DummyCoding(; levels=levs)) == levs

        @test T.baselevel(T.DummyCoding(; levels=levs, base=base)) == base
        @test levels(T.DummyCoding(; levels=levs, base=base)) == levs

        should_equal = [
            T.ContrastsMatrix(T.DummyCoding(), ["1", "2", "3", "4"]),
            T.ContrastsMatrix(T.DummyCoding(; base="1"), ["1", "2", "3", "4"]),
            T.ContrastsMatrix(
                T.DummyCoding(; base="1", levels=["1", "2", "3", "4"]), ["1", "2", "3", "4"]
            ),
            T.ContrastsMatrix(
                T.DummyCoding(; levels=["1", "2", "3", "4"]), ["1", "2", "3", "4"]
            ),
        ]

        should_not_equal = [
            T.ContrastsMatrix(T.DummyCoding(), ["2", "3", "4"]),
            T.ContrastsMatrix(T.DummyCoding(), ["2", "3", "4", "1"]),
            T.ContrastsMatrix(T.DummyCoding(; base="2"), ["1", "2", "3", "4"]),
            T.ContrastsMatrix(
                T.DummyCoding(; base="1", levels=["2", "1", "3", "4"]), ["1", "2", "3", "4"]
            ),
        ]

        f = @formula y_int ~ x_cat
        sch = T.schema(f, nt_str)
        ts = T.apply_schema(f.rhs, sch)

        for c in should_equal
            @test sch.schema[T.term(:x_cat)].contrasts == c
            @test hash(sch.schema[T.term(:x_cat)].contrasts) == hash(c)
        end

        for c in should_not_equal
            @test sch.schema[T.term(:x_cat)].contrasts != c
        end
    end

    @testset "FullDummyCoding" begin
        levs = [:a, :b, :c, :d]
        base = [:c]

        @test T.baselevel(T.FullDummyCoding()) == nothing
        @test levels(T.FullDummyCoding()) == nothing
        @test_throws MethodError T.FullDummyCoding(; levels=levs)
        @test_throws MethodError T.FullDummyCoding(; base=base)
    end

    @testset "Non-unique levels" begin
        @test_throws ArgumentError T.ContrastsMatrix(T.DummyCoding(), ["a", "a", "b"])
    end
end
