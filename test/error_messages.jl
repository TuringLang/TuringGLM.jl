@testset "error_messages" begin
    using TuringGLM: concrete_term, schema
    using TuringGLM: Term
    d = (yyy=rand(10), aaa=rand(10), bbb=repeat([:a, :b], 5))
    @test_throws ArgumentError(
        "There isn't a variable called 'aa' in your data; the nearest names appear to be: aaa",
    ) concrete_term(Term(:aa), d, nothing)
    @test_throws ArgumentError(
        "There isn't a variable called 'aab' in your data; the nearest names appear to be: aaa, bbb",
    ) concrete_term(Term(:aab), d, nothing)

    @test_throws ArgumentError("Column 'a' is empty.") concrete_term(
        Term(:a), (; a=[]), nothing
    )
    @test_throws ArgumentError("Column 'a' is empty.") schema((; b=[1, 2], a=[]))
end
