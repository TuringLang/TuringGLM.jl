@testset "error_messages" begin
    d = (yyy=rand(10), aaa=rand(10), bbb=repeat([:a, :b], 5))
    @test_throws ArgumentError(
        "There isn't a variable called 'aa' in your data; the nearest names appear to be: aaa",
    ) T.concrete_term(T.Term(:aa), d, nothing)
    @test_throws ArgumentError(
        "There isn't a variable called 'aab' in your data; the nearest names appear to be: aaa, bbb",
    ) T.concrete_term(T.Term(:aab), d, nothing)

    @test_throws ArgumentError("Column 'a' is empty.") T.concrete_term(
        T.Term(:a), (; a=[]), nothing
    )
    @test_throws ArgumentError("Column 'a' is empty.") T.schema((; b=[1, 2], a=[]))
end
