abstract type AbstractTerm end
const TermOrTerms = Union{AbstractTerm,Tuple{AbstractTerm,Vararg{AbstractTerm}}}
const TupleTerm = Tuple{TermOrTerms,Vararg{TermOrTerms}}

function width(::T) where {T<:AbstractTerm}
    return throw(ArgumentError("terms of type $T have undefined width"))
end

"""
    Term <: AbstractTerm

A placeholder for a variable in a formula where the type (and necessary data
invariants) is not yet known.  This will be converted to a
[`TuringGLM.ContinuousTerm`](@ref) or [`TuringGLM.CategoricalTerm`](@ref)
by [`TuringGLM.apply_schema`](@ref).

# Fields

* `sym::Symbol`: The name of the data column this term refers to.
"""
struct Term <: AbstractTerm
    sym::Symbol
end

function width(::Term)
    return throw(
        ArgumentError(
            "Un-typed Terms have undefined width.  " *
            "Did you forget to call apply_schema?",
        ),
    )
end

"""
    FormulaTerm{L,R} <: AbstractTerm

Represents an entire formula, with a left- and right-hand side.  These can be of
any type (captured by the type parameters).

# Fields

* `lhs::L`: The left-hand side (e.g., response)
* `rhs::R`: The right-hand side (e.g., predictors)
"""
struct FormulaTerm{L,R} <: AbstractTerm
    lhs::L
    rhs::R
end

"""
    InteractionTerm{Ts} <: AbstractTerm

Represents an _interaction_ between two or more individual terms.
Generated by combining multiple `AbstractTerm`s with `&` (which is what calls to
`&` in a `@formula` lower to)

# Fields

* `terms::Ts`: the terms that participate in the interaction.
"""
struct InteractionTerm{Ts} <: AbstractTerm
    terms::Ts
end

width(ts::InteractionTerm) = prod(width(t) for t in ts.terms)

"""
    ContinuousTerm <: AbstractTerm

Represents a continuous variable, with a name and summary statistics.

# Fields

* `sym::Symbol`: The name of the variable
* `mean::T`: Mean
* `var::T`: Variance
* `min::T`: Minimum value
* `max::T`: Maximum value
"""
struct ContinuousTerm{T} <: AbstractTerm
    sym::Symbol
    mean::T
    var::T
    min::T
    max::T
end

width(::ContinuousTerm) = 1

"""
    CategoricalTerm{C,T,N} <: AbstractTerm
Represents a categorical term, with a name and [`TuringGLM.ContrastsMatrix`](@ref)

# Fields

* `sym::Symbol`: The name of the variable
* `contrasts::ContrastsMatrix`: A contrasts matrix that captures the unique
  values this variable takes on and how they are mapped onto numerical
  predictors.
"""
struct CategoricalTerm{C,T,N} <: AbstractTerm
    sym::Symbol
    contrasts::ContrastsMatrix{C,T}
end

width(::CategoricalTerm{C,T,N}) where {C,T,N} = N

# constructor that computes the width based on the contrasts matrix
function CategoricalTerm(sym::Symbol, contrasts::ContrastsMatrix{C,T}) where {C,T}
    return CategoricalTerm{C,T,length(contrasts.termnames)}(sym, contrasts)
end

"""
    MatrixTerm{Ts} <: AbstractTerm

A collection of terms that should be combined to produce a single numeric matrix.

A matrix term is created by [`TuringGLM.data_fixed_effects`](@ref) from a tuple of terms using
[`TuringGLM.collect_matrix_terms`](@ref), which pulls out all the terms that are matrix
terms as determined by the trait function [`TuringGLM.is_matrix_term`](@ref), which is
true by default for all `AbstractTerm`s.
"""
struct MatrixTerm{Ts<:TupleTerm} <: AbstractTerm
    terms::Ts
end

# wrap single terms in a tuple
MatrixTerm(t::AbstractTerm) = MatrixTerm((t,))
width(t::MatrixTerm) = sum(width(tt) for tt in t.terms)

"""
    collect_matrix_terms(ts::TupleTerm)
    collect_matrix_terms(t::AbstractTerm) = collect_matrix_term((t, ))

Depending on whether the component terms are matrix terms (meaning they have
[`is_matrix_term(T) == true`](@ref is_matrix_term)), `collect_matrix_terms` will
return

1.  A single `MatrixTerm` (if all components are matrix terms)
2.  A tuple of the components (if none of them are matrix terms)
3.  A tuple of terms, with all matrix terms collected into a single `MatrixTerm`
    in the first element of the tuple, and the remaining non-matrix terms passed
    through unchanged.

By default all terms are matrix terms (that is,
`is_matrix_term(::Type{<:AbstractTerm}) = true`), the first case is by far the
most common.  The others are provided only for convenience when dealing with
specialized terms that can't be concatenated into a single model matrix, like
random-effects terms.
"""
function collect_matrix_terms(ts::TupleTerm)
    ismat = collect(is_matrix_term.(ts))
    if all(ismat)
        MatrixTerm(ts)
    elseif any(ismat)
        matterms = ts[ismat]
        (MatrixTerm(ts[ismat]), ts[.!ismat]...)
    else
        ts
    end
end

function collect_matrix_terms(t::T) where {T<:AbstractTerm}
    return is_matrix_term(T) ? MatrixTerm((t,)) : t
end
collect_matrix_terms(t::MatrixTerm) = t

"""
    is_matrix_term(::Type{<:AbstractTerm})

Does this type of term get concatenated with other matrix terms into a single
model matrix?  This controls the behavior of the [`TuringGLM.collect_matrix_terms`](@ref),
which collects all of its arguments for which `is_matrix_term` returns `true`
into a [`TuringGLM.MatrixTerm`](@ref), and returns the rest unchanged.

Since all "normal" terms which describe one or more model matrix columns are
matrix terms, this defaults to `true` for any `AbstractTerm`.
An example of a non-matrix term is a random-effect term
[`TuringGLM.RandomEffectTerm`](@ref).
"""
is_matrix_term(::T) where {T} = is_matrix_term(T)
is_matrix_term(::Type{<:AbstractTerm}) = true

extract_symbols(x) = Symbol[]
extract_symbols(x::Symbol) = [x]
function extract_symbols(ex::Expr)
    return is_call(ex) ? mapreduce(extract_symbols, union, ex.args[2:end]) : Symbol[]
end

################################################################################
# showing terms

function Base.show(io::IO, mime::MIME"text/plain", term::AbstractTerm; prefix="")
    return print(io, prefix, term)
end

function Base.show(io::IO, mime::MIME"text/plain", terms::TupleTerm; prefix=nothing)
    for t in terms
        show(io, mime, t; prefix=something(prefix, ""))
        # ensure that there are newlines in between each term after the first
        # if no prefix is specified
        prefix = something(prefix, '\n')
    end
end
Base.show(io::IO, terms::TupleTerm) = join(io, terms, " + ")

function Base.show(io::IO, ::MIME"text/plain", t::Term; prefix="")
    return print(io, prefix, t.sym, "(unknown)")
end
Base.show(io::IO, t::Term) = print(io, t.sym)

Base.show(io::IO, t::FormulaTerm) = print(io, "$(t.lhs) ~ $(t.rhs)")
function Base.show(io::IO, mime::MIME"text/plain", t::FormulaTerm; prefix="")
    println(io, "FormulaTerm")
    print(io, "Response:")
    show(io, mime, t.lhs; prefix="\n  ")
    println(io)
    print(io, "Predictors:")
    return show(io, mime, t.rhs; prefix="\n  ")
end

Base.show(io::IO, it::InteractionTerm) = join(io, it.terms, " & ")
function Base.show(io::IO, mime::MIME"text/plain", it::InteractionTerm; prefix="")
    for t in it.terms
        show(io, mime, t; prefix=prefix)
        prefix = " & "
    end
end

Base.show(io::IO, t::ContinuousTerm) = print(io, t.sym)
function Base.show(io::IO, ::MIME"text/plain", t::ContinuousTerm; prefix="")
    return print(io, prefix, t.sym, "(continuous)")
end

Base.show(io::IO, t::CategoricalTerm{C,T,N}) where {C,T,N} = print(io, t.sym)
function Base.show(
    io::IO, ::MIME"text/plain", t::CategoricalTerm{C,T,N}; prefix=""
) where {C,T,N}
    return print(io, prefix, t.sym, "($C:$(length(t.contrasts.levels))→$N)")
end

Base.show(io::IO, t::MatrixTerm) = show(io, t.terms)
function Base.show(io::IO, mime::MIME"text/plain", t::MatrixTerm; prefix="")
    return show(io, mime, t.terms; prefix=prefix)
end

################################################################################
# operators on Terms that create new terms:

Base.:~(lhs::TermOrTerms, rhs::TermOrTerms) = FormulaTerm(lhs, rhs)

Base.:&(terms::AbstractTerm...) = InteractionTerm(terms)
Base.:&(term::AbstractTerm) = term
function Base.:&(it::InteractionTerm, terms::AbstractTerm...)
    return InteractionTerm((it.terms..., terms...))
end

Base.:+(a::AbstractTerm) = a
Base.:+(a::AbstractTerm, b::AbstractTerm) = a == b ? a : (a, b)
Base.:+(as::TupleTerm, b::AbstractTerm) = b in as ? as : (as..., b)
Base.:+(a::AbstractTerm, bs::TupleTerm) = a in bs ? bs : (a, bs...)
Base.:+(as::TupleTerm, bs::TupleTerm) = (union(as, bs)...,)

"""
    coefnames(term::AbstractTerm)

Return the name(s) of column(s) generated by a term.  Return value is either a
`String` or an iterable of `String`s.
"""
coefnames(t::FormulaTerm) = (coefnames(t.lhs), coefnames(t.rhs))
coefnames(t::ContinuousTerm) = string(t.sym)
function coefnames(t::CategoricalTerm)
    return ["$(t.sym): $name" for name in t.contrasts.termnames]
end
coefnames(ts::TupleTerm) = reduce(vcat, coefnames.(ts))
coefnames(t::MatrixTerm) = mapreduce(coefnames, vcat, t.terms)
function coefnames(t::InteractionTerm)
    return kron_insideout(
        (args...) -> join(args, " & "), vectorize.(coefnames.(t.terms))...
    )
end

hasresponse(t) = false
hasresponse(t::FormulaTerm) = t.lhs !== nothing

"""
    term(x)

Wrap argument in an appropriate `AbstractTerm` type: `Symbol`s and `AbstractString`s
become `Term`s. Any `AbstractTerm`s are unchanged. `AbstractString`s are converted to
symbols before wrapping.
"""
term(s::Symbol) = Term(s)
term(s::AbstractString) = term(Symbol(s))
term(t::AbstractTerm) = t
