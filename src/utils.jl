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
    return vec(μ_X), X_centered
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
    return vec(μ_X), vec(σ_X), X_std
end

"""
    standardize_predictors(x::AbstractVector)

Standardizes the vector `x` to mean 0 and standard deviation 1.

Returns a tuple with:
1. `μ_X`: `Float64`s of the mean of the original vector `x`.
2. `σ_X`: `Float64`s of the standard deviations of the original vector `x`.
3. `x_std`: A `Vector` of `Float64`s with the same length as the original vector
`x` with the values centered on mean μ=0 and standard deviation σ=1.

# Arguments
- `x::AbstractVector`: a vector.
"""
function standardize_predictors(x::AbstractVector)
    μ_x = mean(x)
    σ_x = std(x)
    x_std = (x .- μ_x) ./ σ_x
    return μ_x, σ_x, x_std
end

"""
    tuple_length(::NTuple{N, Any}) where {N} = Int(N)

This is a hack to get the length of any tuple.
"""
tuple_length(::NTuple{N,Any}) where {N} = Int(N)

"""
    convert_str_to_indices(v::AbstractVector)

Converts a vector `v` to a vector of indices, i.e. a vector where all the entries are
integers. Returns a tuple with the first element as the converted vector and the
second element a `Dict` specifying which string is which integer.

This function is especially useful for random-effects varying-intercept hierarchical models.
Normally `v` would be a vector of group membership with values such as `"group_1"`,
`"group_2"` etc. For random-effect models with varying-intercepts, Turing needs the group
membership values to be passed as `Int`s.
"""
function convert_str_to_indices(v::AbstractVector)
    d = Dict{eltype(v),Int}()
    v_int = Int[]
    for i in v
        n = get!(d, i, length(d) + 1)
        push!(v_int, n)
    end
    return v_int, d
end
