using Tables
using Statistics: mean
using StatsModels:
    @formula, hasintercept, modelcols, FormulaTerm, apply_schema, schema, coefnames

"""
    center_predictors(X::AbstractMatrix)

Centers the columns of a matrix `X` of predictors.
Returns a tuple with:
1. `μ_X`: 1xK `Matrix` of `Float64`s of the means of the K columns in the original `X`
matrix.
2. `X_centered`: A `Matrix` of `Float64`s with the same dimensions as the original matrix
`X` with the columns centered on mean μ=0.

# Arguments
- `X::AbstractMatrix`: a matrix of predictors where rows are observations and columns are
variables.

# Examples
```jldoctest
julia> X = rand(3, 2)
3×2 Matrix{Float64}:
 0.291245   0.604541
 0.0960436  0.609567
 0.536451   0.0184553

julia> μ_X, X_centered = center_predictors(X);

julia> μ_X
1×2 Matrix{Float64}:
 0.307913  0.410854

julia> X_centered
3×2 Matrix{Float64}:
 -0.0166683   0.193686
 -0.21187     0.198713
  0.228538   -0.392399
```
"""
function center_predictors(X::AbstractMatrix)
    μ_X = mapslices(mean, X; dims=1)
    X_centered = similar(X, Float64)
    for (idx, val) in enumerate(eachcol(X))
        X_centered[:, idx] = val .- μ_X[1, idx]
    end
    return μ_X, X_centered
end

"""
    make_yX(formula::FormulaTerm, data)

Constructs the response y vector and the matrix X of predictors.

Note that the original `StatsModels.jl` implementation adds an intercept column filled
with `1`s, which we do not.

Returns a tuple with:
1. `y`: A `Vector` of the response variable in the `formula` and present inside `data`.
2. `X`: A `Matrix` of the predictors variables in the `formula` and present inside `data`.

# Arguments
- `formula`: a `FormulaTerm` created by `StatsModels.@formula` macro.
- `data`: `data` can either be a whole table (a property-accessible collection of iterable
columns or iterable collection of property-accessible rows, as defined by
[Tables.jl](https://github.com/JuliaData/Tables.jl), which includes `DataFrame`s) or a
single row (in the form of a `NamedTuple` of scalar values).

# Examples
```jldoctest
julia> using CategoricalArrays

julia> using DataFrames

julia> df = DataFrame(;
           y_int=[2, 3, 4, 5],
           x_float=[1.1, 2.3, 3.14, 3.65],
           x_cat=categorical([1, 2, 3, 4]),
       )
4×3 DataFrame
 Row │ y_int  x_float  x_cat
     │ Int64  Float64  Cat…
─────┼───────────────────────
   1 │     2     1.1   1
   2 │     3     2.3   2
   3 │     4     3.14  3
   4 │     5     3.65  4


julia> formula = @formula y_int ~ 1 + x_float + x_cat;

julia> y, X = make_yX(formula, df)
([2, 3, 4, 5], [1.1 0.0 0.0 0.0; 2.3 1.0 0.0 0.0; 3.14 0.0 1.0 0.0; 3.65 0.0 0.0 1.0])

julia> y
4-element Vector{Int64}:
 2
 3
 4
 5

julia> X
4×4 Matrix{Float64}:
 1.1   0.0  0.0  0.0
 2.3   1.0  0.0  0.0
 3.14  0.0  1.0  0.0
 3.65  0.0  0.0  1.0
```
"""
function make_yX(formula::FormulaTerm, data::D) where {D}
    Tables.istable(data) || throw(ArgumentError("Data of type $D is not a table!"))
    ts = apply_schema(formula, schema(data))
    y, X = modelcols(ts, df)
    if hasintercept(formula)
        X = X[:, 2:end]
    end
    return y, X
end

