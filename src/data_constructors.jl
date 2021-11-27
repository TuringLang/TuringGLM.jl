"""
    modelcols(t::AbstractTerm, data)

Create a numerical "model columns" representation of data based on an
`AbstractTerm`.  `data` can either be a whole table (a property-accessible
collection of iterable columns or iterable collection of property-accessible
rows, as defined by [Tables.jl](https://github.com/JuliaData/Tables.jl) or a
single row (in the form of a `NamedTuple` of scalar values).
"""
modelcols(t::ContinuousTerm, d::NamedTuple) = copy.(d[t.sym])
modelcols(t::CategoricalTerm, d::NamedTuple) = t.contrasts[d[t.sym], :]
function modelcols(t::MatrixTerm, d::Tables.ColumnTable)
    mat = reduce(hcat, [modelcols(tt, d) for tt in t.terms])
    return reshape(mat, size(mat, 1), :)
end

modelcols(t::MatrixTerm, d::NamedTuple) = reduce(vcat, [modelcols(tt, d) for tt in t.terms])
# two options here: either special-case ColumnTable (named tuple of vectors)
# vs. vanilla NamedTuple, or reshape and use normal broadcasting
function modelcols(t::InteractionTerm, d::NamedTuple)
    return kron_insideout(*, (modelcols(term, d) for term in t.terms)...)
end

function modelcols(t::InteractionTerm, d::Tables.ColumnTable)
    return row_kron_insideout(*, (modelcols(term, d) for term in t.terms)...)
end

vectorize(x::Tuple) = collect(x)
vectorize(x::AbstractVector) = x
vectorize(x) = [x]

"""
    reshape_last_to_i(i::Int, a)

Reshape `a` so that its last dimension moves to dimension `i` (+1 if `a` is an
`AbstractMatrix`).
"""
reshape_last_to_i(i, a) = a
reshape_last_to_i(i, a::AbstractVector) = reshape(a, ones(Int, i - 1)..., :)
reshape_last_to_i(i, a::AbstractMatrix) = reshape(a, size(a, 1), ones(Int, i - 1)..., :)

# an "inside out" kronecker-like product based on broadcasting reshaped arrays
# for a single row, some will be scalars, others possibly vectors.  for a whole
# table, some will be vectors, possibly some matrices
function kron_insideout(op::Function, args...)
    args = (reshape_last_to_i(i, a) for (i, a) in enumerate(args))
    out = broadcast(op, args...)
    # flatten array output to vector
    return out isa AbstractArray ? vec(out) : out
end

function row_kron_insideout(op::Function, args...)
    rows = size(args[1], 1)
    args = (reshape_last_to_i(i, reshape(a, size(a, 1), :)) for (i, a) in enumerate(args))
    # args = (reshape(a, size(a,1), ones(Int, i-1)..., :) for (i,a) in enumerate(args))
    return reshape(broadcast(op, args...), rows, :)
end

"""
    data_response(formula::FormulaTerm, data)

Constructs the response y vector.

Returns a `Vector` of the response variable in the `formula` and present inside `data`.
# Arguments
- `formula`: a `FormulaTerm` created by `@formula` macro.
- `data`:  a `data` object that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface such as a DataFrame.
"""
function data_response(formula::FormulaTerm, data::D) where {D}
    Tables.istable(data) || throw(ArgumentError("Data of type $D is not a table!"))
    sch = schema(formula, data)
    ts = apply_schema(formula.lhs, sch)
    y = modelcols(ts, Tables.columntable(data))
    return y
end

"""
    data_fixed_effects(formula::FormulaTerm, data)

Constructs the matrix X of fixed-effects (a.k.a. population-level) predictors.

Returns a `Matrix` of the fixed-effects predictors variables in the `formula`
and present inside `data`.

# Arguments
- `formula`: a `FormulaTerm` created by `@formula` macro.
- `data`:  a `data` object that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface such as a DataFrame.
"""
function data_fixed_effects(formula::FormulaTerm, data::D) where {D}
    Tables.istable(data) || throw(ArgumentError("Data of type $D is not a table!"))
    sch = schema(formula, data)
    ts = apply_schema(formula.rhs, sch)
    ts = collect_matrix_terms(ts)
    X = modelcols(ts, Tables.columntable(data))
    return X
end

"""
    data_random_effects(formula::FormulaTerm, data)

Constructs the matrix(ces) Z(s) of random-effects (a.k.a. group-level) predictors.

Returns a Tuple of `Matrix` of the random-effects predictors variables in the `formula`
and present inside `data`.

# Arguments
- `formula`: a `FormulaTerm` created by `@formula` macro.
- `data`:  a `data` object that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface such as a DataFrame.
"""
function data_random_effects(formula::FormulaTerm, data::D) where {D}
    Tables.istable(data) || throw(ArgumentError("Data of type $D is not a table!"))
    Z = nothing
    # TODO:
    # is_matrix_terms false for random-effects.
    return Z
end

function _has_ranef(formula::FormulaTerm, data)
    return false
end
