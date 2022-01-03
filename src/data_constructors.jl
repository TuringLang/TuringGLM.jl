mutable struct SlopePerRanEf
    grouping_vars::Dict{String,Vector{String}}
    # inner constructors
    SlopePerRanEf() = new(Dict{String,Vector{String}}())
    SlopePerRanEf(d::Dict{String,Vector{String}}) = new(d)
end
isequal(x::SlopePerRanEf, y::SlopePerRanEf) = x.grouping_vars == y.grouping_vars
==(x::SlopePerRanEf, y::SlopePerRanEf) = x.grouping_vars == y.grouping_vars
length(x::SlopePerRanEf) = length(values(x.grouping_vars))

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

Constructs the vector(s)/matrix(ces) Z(s) of random-effects (a.k.a. group-level)
slope predictors.

Returns a `Dict{String, AbstractArray}` of `Vector`/`Matrix` as values of the random-effects
predictors slope variables (keys) in the `formula` and present inside `data`.

# Arguments
- `formula`: a `FormulaTerm` created by `@formula` macro.
- `data`:  a `data` object that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface such as a DataFrame.
"""
function data_random_effects(formula::FormulaTerm, data::D) where {D}
    if !has_ranef(formula)
        return nothing
    end
    slopes = slope_per_ranef(ranef(formula))

    Z = Dict{String,AbstractArray}() # empty Dict
    if length(slopes) > 0
        # add the slopes to Z
        # this would need to create a vector from the column of the X matrix from the
        # slope term
        for slope in values(slopes.grouping_vars)
            if slope isa String
                Z["slope_" * slope] = get_var(term(slope), data)
            else
                for s in slope
                    Z["slope_" * s] = get_var(term(s), data)
                end
            end
        end
    else
        Z = nothing
    end
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
        terms = filter(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
        terms = map(terms) do t
            lhs, rhs = first(t.args_parsed), last(t.args_parsed)
            RandomEffectsTerm(lhs, rhs)
        end
        return terms
    else
        return nothing
    end
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

Returns a `SlopePerRanEf` object where the entries are the grouping variables that have
a group-level slope.
"""
function slope_per_ranef(terms::Tuple)
    slopes = SlopePerRanEf()
    for ts in terms
        if ts.lhs isa Term
            slopes.grouping_vars[string(ts.rhs)] = [string(ts.rhs)]
        elseif ts.lhs isa Tuple && any(t -> t isa Term, ts.lhs)
            # empty first
            slopes.grouping_vars[string(ts.rhs)] = Vector{String}()
            # now populate
            for tleft in ts.lhs
                if tleft isa Term
                    push!(slopes.grouping_vars[string(ts.rhs)], string(tleft))
                end
            end
        end
    end
    return slopes
end

"""
    get_idx(term::Term, data)

Returns a tuple with the first element as the ID vector of `Int`s that represent
group membership for a specific random-effect intercept group `t` of observations
present in `data`. The second element of the tuple is a `Dict` specifying which string is
which integer in the ID vector.
"""
function get_idx(t::Term, data::D) where {D}
    col = Symbol(t)
    idx = Tables.getcolumn(data, col)
    return convert_str_to_indices(idx)
end

"""
    get_var(term::Term, data)

Returns the corresponding vector of column in `data` for the a specific
random-effect slope `term` of observations present in `data`.
"""
function get_var(t::Term, data::D) where {D}
    col = Symbol(t)
    return Tables.getcolumn(data, col)
end
