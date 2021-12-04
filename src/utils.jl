"""
    center_predictors(X::AbstractMatrix)

Centers the columns of a matrix `X` of predictors to mean 0.

Returns a tuple with:
1. `μ_X`: 1xK `Matrix` of `Float64`s of the means of the K columns in the original `X`
matrix.
2. `X_centered`: A `Matrix` of `Float64`s with the same dimensions as the original matrix
`X` with the columns centered on mean μ=0.

# Arguments
- `X::AbstractMatrix`: a matrix of predictors where rows are observations and columns are
variables.
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
    standardize_predictors(X::AbstractMatrix)

Standardizes the columns of a matrix `X` of predictors to mean 0 and standard deviation 1.

Returns a tuple with:
1. `μ_X`: 1xK `Matrix` of `Float64`s of the means of the K columns in the original `X`
matrix.
2. `σ_X`: 1xK `Matrix` of `Float64`s of the standard deviations of the K columns in the
original `X` matrix.
3. `X_std`: A `Matrix` of `Float64`s with the same dimensions as the original matrix
`X` with the columns centered on mean μ=0 and standard deviation σ=1.

# Arguments
- `X::AbstractMatrix`: a matrix of predictors where rows are observations and columns are
variables.
"""
function standardize_predictors(X::AbstractMatrix)
    μ_X = mapslices(mean, X; dims=1)
    σ_X = mapslices(std, X; dims=1)
    X_std = similar(X, Float64)
    for (idx, val) in enumerate(eachcol(X))
        X_std[:, idx] = (val .- μ_X[1, idx]) ./ σ_X[1, idx]
    end
    return μ_X, σ_X, X_std
end

"""
    tuple_length(::NTuple{N, Any}) where {N} = Int(N)

This is a hack to get the length of any tuple.
"""
tuple_length(::NTuple{N,Any}) where {N} = Int(N)

