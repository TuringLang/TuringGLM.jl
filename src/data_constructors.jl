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
    y = response(formula, data)
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
    if _has_ranef(formula)
        X = MixedModels.modelmatrix(MixedModel(formula, data))
        X = X[:, 2:end]
    else
        X = StatsModels.modelmatrix(formula, data)
        if hasintercept(formula)
            X = X[:, 2:end]
        end
    end
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
    Z = nothing
    # TODO:
    # is_matrix_terms false for random-effects.
    return Z
end

function _has_ranef(formula::FormulaTerm)
    return any(t -> t isa FunctionTerm, formula.rhs)
end
