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
    if has_ranef(formula)
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

"""
    has_ranef(formula::FormulaTerm)

Returns `true` if any of the terms in `formula` is a `FunctionTerm` or false
otherwise.
"""
function has_ranef(formula::FormulaTerm)
    return any(t -> t isa FunctionTerm, formula.rhs)
end

"""
    ranef(formula::FormulaTerm)

Returns a tuple of the `FunctionTerm`s parsed as `RandomEffectsTerm`s in `formula`.
If there are no `FunctionTerm`s in `formula` returns `nothing`.
"""
function ranef(formula::FormulaTerm)
    if has_ranef(formula)
        terms = filter(t -> t isa FunctionTerm, formula.rhs)
        terms = map(
            t -> RandomEffectsTerm(first(t.args_parsed), last(t.args_parsed)), terms
        )
    else
        terms = nothing
    end
    return terms
end

"""
    n_ranef(formula::FormulaTerm)

Returns the number of `RandomEffectsTerm`s in `formula`.
"""
function n_ranef(formula::FormulaTerm)
    terms = ranef(formula)
    if terms === nothing
        return 0
    elseif any(t -> t isa RandomEffectsTerm, terms)
        counter = 0
        for ts in terms
            if ts.lhs isa Tuple
                counter += tuple_length(ts.lhs)
            else
                counter += 1
            end
        end
        return counter
    else
        # fallback
        return 1
    end
end

"""
    intercept_per_ranef(terms::Tuple{RandomEffectsTerm})

Returns a vector of `String`s where the entries are the grouping variables that have
a group-level intercept.
"""
function intercept_per_ranef(terms::Tuple)
    vec_intercepts = Vector{String}()
    for ts in terms
        if ts.lhs isa ConstantTerm
            push!(vec_intercepts, string(ts.rhs))
        elseif ts.lhs isa Tuple && any(t -> t isa ConstantTerm, ts.lhs)
            push!(vec_intercepts, string(ts.rhs))
        end
    end
    return vec_intercepts
end

"""
    slope_per_ranef(terms::Tuple{RandomEffectsTerm})

Returns a vector of `String`s where the entries are the grouping variables that have
a group-level slope.
"""
function slope_per_ranef(terms::Tuple)
    vec_slopes = Vector{String}()
    for ts in terms
        if ts.lhs isa Term
            push!(vec_slopes, string(ts.rhs))
        elseif ts.lhs isa Tuple && any(t -> t isa Term, ts.lhs)
            push!(vec_slopes, string(ts.rhs))
        end
    end
    return vec_slopes
end
