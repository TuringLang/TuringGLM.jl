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
    model_response(formula::StatsModels.FormulaTerm, data)

Constructs the response y vector.

Returns a `Vector` of the response variable in the `formula` and present inside `data`.
# Arguments
- `formula`: a `FormulaTerm` created by `StatsModels.@formula` macro.
- `data`:  a `data` object that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface such as a DataFrame.
"""
function model_response(formula::StatsModels.FormulaTerm, data::D) where {D}
    Tables.istable(data) || throw(ArgumentError("Data of type $D is not a table!"))
    ts = StatsModels.apply_schema(
        formula, StatsModels.schema(data), MixedModels.LinearMixedModel
    )
    y = first(StatsModels.modelcols(ts, data))
    return y
end

"""
    model_matrix(formula::StatsModels.FormulaTerm, data)

Constructs the matrix X of predictors along with the matrix Z of random effects
(if applicable).

Note that the original `StatsModels.jl` implementation adds an intercept column filled
with `1`s, which we do not.

Returns a`NamedTuple` with:
1. `X`: A `Matrix` of the predictors variables in the `formula` and present inside `data`.
2. `Zs`: One or more random effects matrix (if applicable, otherwise `Zs` is `nothing`).

# Arguments
- `formula`: a `FormulaTerm` created by `StatsModels.@formula` macro.
- `data`:  a `data` object that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface such as a DataFrame.
"""
function model_matrix(formula::StatsModels.FormulaTerm, data::D) where {D}
    Tables.istable(data) || throw(ArgumentError("Data of type $D is not a table!"))
    ts = StatsModels.apply_schema(formula, StatsModels.schema(data), MixedModels.MixedModel)
    if _has_ranef(ts, data)
        _, Xs = StatsModels.modelcols(ts, data)
        X = first(Xs)
        Zs = last(Xs)
    elseif !(_has_ranef(ts, data))
        _, X = StatsModels.modelcols(ts, data)
        Zs = nothing
    end
    if StatsModels.hasintercept(formula)
        X = X[:, 2:end]
    end
    return (; X, Zs)
end

# I have no idea how to test if a `formula` has MixedModels.RandomEffectsTerms.
# So this is a hack, if the return value from `modelmatrix` is a Tuple
# (hence not an AbstractMatrix), we know that we have a RandomEffectsTerms in our hands.
function _has_ranef(formula::StatsModels.FormulaTerm, data)
    return !(MixedModels.modelmatrix(formula, data) isa AbstractMatrix)
end
