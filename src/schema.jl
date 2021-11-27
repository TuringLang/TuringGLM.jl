################################################################################
# Schemas for terms

# step 1: extract all Term symbols
# step 2: create empty Schema (Dict)
# step 3: for each term, create schema entry based on column from data store

terms(t::FormulaTerm) = union(terms(t.lhs), terms(t.rhs))
terms(t::InteractionTerm) = terms(t.terms)
terms(t::AbstractTerm) = [t]
terms(t::MatrixTerm) = terms(t.terms)
terms(t::TupleTerm) = mapreduce(terms, union, t)

needs_schema(::AbstractTerm) = true
needs_schema(::ConstantTerm) = false
needs_schema(t) = false
needs_schema(::Union{CategoricalTerm,ContinuousTerm,InterceptTerm}) = false

"""
    Schema

Struct that wraps a `Dict` mapping `Term`s to their concrete forms.  This exists
mainly for dispatch purposes and to support possibly more sophisticated behavior
in the future.

A `Schema` behaves for all intents and purposes like an immutable `Dict`, and
delegates the constructor as well as `getindex`, `get`, `merge!`, `merge`,
`keys`, and `haskey` to the wrapped `Dict`.
"""
struct Schema
    schema::Dict{Term,AbstractTerm}
    Schema(x...) = new(Dict{Term,AbstractTerm}(x...))
end

function Base.show(io::IO, schema::Schema)
    n = length(schema.schema)
    println(io, "TuringGLM.Schema with $n ", n == 1 ? "entry:" : "entries:")
    for (k, v) in schema.schema
        println(io, "  ", k, " => ", v)
    end
end

Base.getindex(schema::Schema, key) = getindex(schema.schema, key)
Base.get(schema::Schema, key, default) = get(schema.schema, key, default)
Base.merge(a::Schema, b::Schema) = Schema(merge(a.schema, b.schema))
Base.merge!(a::Schema, b::Schema) = (merge!(a.schema, b.schema); a)

Base.keys(schema::Schema) = keys(schema.schema)
Base.haskey(schema::Schema, key) = haskey(schema.schema, key)

"""
    schema([terms::AbstractVector{<:AbstractTerm}, ]data, hints::Dict{Symbol})
    schema(term::AbstractTerm, data, hints::Dict{Symbol})

Compute all the invariants necessary to fit a model with `terms`.  A schema is a dict that
maps `Term`s to their concrete instantiations (either `CategoricalTerm`s or
`ContinuousTerm`s.  "Hints" may optionally be supplied in the form of a `Dict` mapping term
names (as `Symbol`s) to term or contrast types.  If a hint is not provided for a variable,
the appropriate term type will be guessed based on the data type from the data column: any
numeric data is assumed to be continuous, and any non-numeric data is assumed to be
categorical.

Return a [`TuringGLM.Schema`](@ref), which is a wrapper around a `Dict` mapping `Term`s to
their concrete instantiations (`ContinuousTerm` or `CategoricalTerm`).
"""
schema(data, hints=Dict{Symbol,Any}()) = schema(Tables.columntable(data), hints)
function schema(dt::D, hints=Dict{Symbol,Any}()) where {D<:Tables.ColumnTable}
    return schema(Term.(collect(fieldnames(D))), dt, hints)
end
function schema(ts::AbstractVector{<:AbstractTerm}, data, hints::Dict{Symbol})
    return schema(ts, Tables.columntable(data), hints)
end

# handle hints:
function schema(
    ts::AbstractVector{<:AbstractTerm},
    dt::Tables.ColumnTable,
    hints::Dict{Symbol}=Dict{Symbol,Any}(),
)
    return Schema(t => concrete_term(t, dt, hints) for t in ts)
end

function schema(f::TermOrTerms, data, hints::Dict{Symbol})
    return schema(filter(needs_schema, terms(f)), data, hints)
end

schema(f::TermOrTerms, data) = schema(f, data, Dict{Symbol,Any}())

"""
    concrete_term(t::Term, data[, hint])

Create concrete term from the placeholder `t` based on a data source and
optional hint.  If `data` is a table, the `getproperty` is used to extract the
appropriate column.

The `hint` can be a `Dict{Symbol}` of hints, or a specific hint, a concrete term
type (`ContinuousTerm` or `CategoricalTerm`), or an instance of some
`<:AbstractContrasts`, in which case a `CategoricalTerm` will be created using
those contrasts.

If no hint is provided (or `hint==nothing`), the `eltype` of the data is used:
`Number`s are assumed to be continuous, and all others are assumed to be
categorical.
"""
function concrete_term(t::Term, d, hints::Dict{Symbol})
    return concrete_term(t, d, get(hints, t.sym, nothing))
end

function concrete_term(t::Term, dt::Tables.ColumnTable, hint)
    msg = checkcol(dt, t.sym)
    if msg != ""
        throw(ArgumentError(msg))
    end
    return concrete_term(t, getproperty(dt, t.sym), hint)
end

function concrete_term(t::Term, dt::Tables.ColumnTable, hints::Dict{Symbol})
    msg = checkcol(dt, t.sym)
    if msg != ""
        throw(ArgumentError(msg))
    end
    return concrete_term(t, getproperty(dt, t.sym), get(hints, t.sym, nothing))
end

concrete_term(t::Term, d) = concrete_term(t, d, nothing)

# if the "hint" is already an AbstractTerm, use that
# need this specified to avoid ambiguity
concrete_term(t::Term, d::Tables.ColumnTable, hint::AbstractTerm) = hint
concrete_term(t::Term, x, hint::AbstractTerm) = hint

function concrete_term(t::Term, xs::AbstractVector{<:Number}, ::Nothing)
    return concrete_term(t, xs, ContinuousTerm)
end
function concrete_term(t::Term, xs::AbstractVector, ::Type{ContinuousTerm})
    μ, σ2 = StatsBase.mean_and_var(xs)
    min, max = extrema(xs)
    return ContinuousTerm(t.sym, promote(μ, σ2, min, max)...)
end
# default contrasts: dummy coding
function concrete_term(t::Term, xs::AbstractVector, ::Nothing)
    return concrete_term(t, xs, CategoricalTerm)
end
function concrete_term(t::Term, xs::AbstractArray, ::Type{CategoricalTerm})
    return concrete_term(t, xs, DummyCoding())
end

function concrete_term(t::Term, xs::AbstractArray, contrasts::AbstractContrasts)
    contrmat = ContrastsMatrix(contrasts, intersect(DataAPI.levels(xs), unique(xs)))
    return CategoricalTerm(t.sym, contrmat)
end

"""
    apply_schema(t, schema::Schema)

Return a new term that is the result of applying `schema` to term `t`.

When `t` is a `ContinuousTerm` or `CategoricalTerm` already, the term will be returned
unchanged _unless_ a matching term is found in the schema.  This allows
selective re-setting of a schema to change the contrast coding or levels of a
categorical term, or to change a continuous term to categorical or vice versa.
"""
apply_schema(t, schema) = t
apply_schema(terms::TupleTerm, schema) = reduce(+, apply_schema.(terms, Ref(schema)))

apply_schema(t::Term, schema::Schema) = schema[t]
function apply_schema(it::InteractionTerm, schema::Schema)
    return InteractionTerm(apply_schema(it.terms, schema))
end

# for re-setting schema (in setcontrasts!)
function apply_schema(t::Union{ContinuousTerm,CategoricalTerm}, schema::Schema)
    return get(schema, term(t.sym), t)
end
apply_schema(t::MatrixTerm, sch::Schema) = MatrixTerm(apply_schema.(t.terms, Ref(sch)))

function apply_schema(t::ConstantTerm, schema::Schema, Mod::Type)
    t.n ∈ (-1, 0, 1) || throw(
        ArgumentError(
            "can't create InterceptTerm from $(t.n) " * "(only -1, 0, and 1 allowed)"
        ),
    )
    return InterceptTerm{t.n == 1}()
end

"""
    has_schema(t::T) where {T<:AbstractTerm}

Return `true` if `t` has a schema, meaning that `apply_schema` would be a no-op.
"""
has_schema(t::AbstractTerm) = true
has_schema(t::ConstantTerm) = false
has_schema(t::Term) = false
has_schema(t::Union{ContinuousTerm,CategoricalTerm}) = true
has_schema(t::InteractionTerm) = all(has_schema(tt) for tt in t.terms)
has_schema(t::TupleTerm) = all(has_schema(tt) for tt in t)
has_schema(t::MatrixTerm) = has_schema(t.terms)
has_schema(t::FormulaTerm) = has_schema(t.lhs) && has_schema(t.rhs)

struct FullRank
    schema::Schema
    already::Set{AbstractTerm}
end

FullRank(schema) = FullRank(schema, Set{AbstractTerm}())

Base.get(schema::FullRank, key, default) = get(schema.schema, key, default)
function Base.merge(a::FullRank, b::FullRank)
    return FullRank(merge(a.schema, b.schema), union(a.already, b.already))
end

function apply_schema(t::FormulaTerm, schema::Schema)
    schema = FullRank(schema)

    # only apply rank-promoting logic to RIGHT hand side
    return FormulaTerm(
        apply_schema(t.lhs, schema.schema),
        collect_matrix_terms(apply_schema(t.rhs, schema)),
    )
end

# strategy is: apply schema, then "repair" if necessary (promote to full rank
# contrasts).
#
# to know whether to repair, need to know context a term appears in.  main
# effects occur in "own" context.

"""
    apply_schema(t::AbstractTerm, schema::TuringGLM.FullRank)

Apply a schema, under the assumption that when a less-than-full rank model
matrix would be produced, categorical terms should be "promoted" to full rank
(where a categorical variable with ``k`` levels would produce ``k`` columns,
instead of ``k-1`` in the standard contrast coding schemes).
"""
function apply_schema(t::ConstantTerm, schema::FullRank, Mod::Type)
    push!(schema.already, t)
    return apply_schema(t, schema.schema, Mod)
end

apply_schema(t::InterceptTerm, schema::FullRank, Mod::Type) = (push!(schema.already, t); t)

function apply_schema(t::AbstractTerm, schema::FullRank)
    push!(schema.already, t)
    t = apply_schema(t, schema.schema) # continuous or categorical now
    return apply_schema(t, schema, t) # repair if necessary
end

function apply_schema(t::InteractionTerm, schema::FullRank)
    push!(schema.already, t)
    terms = apply_schema.(t.terms, Ref(schema.schema))
    terms = apply_schema.(terms, Ref(schema), Ref(t))
    return InteractionTerm(terms)
end

# context doesn't matter for non-categorical terms
apply_schema(t, schema::FullRank, context::AbstractTerm) = t
# when there's a context, check to see if any of the terms already seen would be
# aliased by this term _if_ it were full rank.
function apply_schema(t::CategoricalTerm, schema::FullRank, context::AbstractTerm)
    aliased = drop_term(context, t)
    @debug "$t in context of $context: aliases $aliased\n  seen already: $(schema.already)"
    for seen in schema.already
        if symequal(aliased, seen)
            @debug "  aliased term already present: $seen"
            return t
        end
    end
    # aliased term not seen already:
    # add aliased term to already seen:
    push!(schema.already, aliased)
    # repair:
    new_contrasts = ContrastsMatrix(FullDummyCoding(), t.contrasts.levels)
    t = CategoricalTerm(t.sym, new_contrasts)
    @debug "  aliased term absent, repairing: $t"
    return t
end

drop_term(from, to) = symequal(from, to) ? ConstantTerm(1) : from
drop_term(from::FormulaTerm, to) = FormulaTerm(from.lhs, drop_term(from.rhs, to))
drop_term(from::MatrixTerm, to) = MatrixTerm(drop_term(from.terms, to))
drop_term(from::TupleTerm, to) = tuple((t for t = from if !symequal(t, to))...)
function drop_term(from::InteractionTerm, t)
    terms = drop_term(from.terms, t)
    return length(terms) > 1 ? InteractionTerm(terms) : terms[1]
end

"""
    termsyms(t::Terms.Term)

Extract the set of symbols referenced in this term.
This is needed in order to determine when a categorical term should have
standard (reduced rank) or full rank contrasts, based on the context it occurs
in and the other terms that have already been encountered.
"""
termsyms(t::AbstractTerm) = Set()
termsyms(t::InterceptTerm{true}) = Set(1)
termsyms(t::ConstantTerm) = Set((t.n,))
termsyms(t::Union{Term,CategoricalTerm,ContinuousTerm}) = Set([t.sym])
termsyms(t::InteractionTerm) = mapreduce(termsyms, union, t.terms)

symequal(t1::AbstractTerm, t2::AbstractTerm) = issetequal(termsyms(t1), termsyms(t2))

"""
    termvars(t::AbstractTerm)

The data variables that this term refers to.
"""
termvars(::AbstractTerm) = Symbol[]
termvars(t::Union{Term,CategoricalTerm,ContinuousTerm}) = [t.sym]
termvars(t::InteractionTerm) = mapreduce(termvars, union, t.terms)
termvars(t::TupleTerm) = mapreduce(termvars, union, t; init=Symbol[])
termvars(t::MatrixTerm) = termvars(t.terms)
termvars(t::FormulaTerm) = union(termvars(t.lhs), termvars(t.rhs))
