abstract type AbstractContrasts end

# Contrasts + Levels (usually from data) = ContrastsMatrix
struct ContrastsMatrix{C<:AbstractContrasts,T,U}
    matrix::Matrix{Float64}
    termnames::Vector{U}
    levels::Vector{T}
    contrasts::C
    invindex::Dict{T,Int}
    function ContrastsMatrix(
        matrix::AbstractMatrix, termnames::Vector{U}, levels::Vector{T}, contrasts::C
    ) where {U,T,C<:AbstractContrasts}
        allunique(levels) ||
            throw(ArgumentError("levels must be all unique, got $(levels)"))
        invindex = Dict{T,Int}(x => i for (i, x) in enumerate(levels))
        return new{C,T,U}(matrix, termnames, levels, contrasts, invindex)
    end
end

# only check equality of matrix, termnames, and levels, and that the type is the
# same for the contrasts (values are irrelevant).  This ensures that the two
# will behave identically in creating modelmatrix columns
function Base.:(==)(
    a::ContrastsMatrix{C,T}, b::ContrastsMatrix{C,T}
) where {C<:AbstractContrasts,T}
    return a.matrix == b.matrix && a.termnames == b.termnames && a.levels == b.levels
end

function Base.hash(a::ContrastsMatrix{C}, h::UInt) where {C}
    return hash(C, hash(a.matrix, hash(a.termnames, hash(a.levels, h))))
end

"""
An instantiation of a contrast coding system for particular levels

This type is used internally for generating model matrices based on categorical
data, and **most users will not need to deal with it directly**.  Conceptually,
a `ContrastsMatrix` object stands for an instantiation of a contrast coding
*system* for a particular set of categorical *data levels*.

If levels are specified in the `AbstractContrasts`, those will be used, and likewise
for the base level (which defaults to the first level).

# Constructors

```julia
ContrastsMatrix(contrasts::AbstractContrasts, levels::AbstractVector)
ContrastsMatrix(contrasts_matrix::ContrastsMatrix, levels::AbstractVector)
```

# Arguments

* `contrasts::AbstractContrasts`: The contrast coding system to use.
   We only use `DummyCoding()`

* `levels::AbstractVector`: The levels to generate contrasts for.

* `contrasts_matrix::ContrastsMatrix`: Constructing a `ContrastsMatrix` from
  another will check that the levels match.  This is used, for example, in
  constructing a model matrix from a `ModelFrame` using different data.
"""
function ContrastsMatrix(
    contrasts::C, levels::AbstractVector{T}
) where {C<:AbstractContrasts,T}

    # if levels are defined on contrasts, use those, validating that they line up.
    # what does that mean? either:
    #
    # 1. DataAPI.levels(contrasts) == levels (best case)
    # 2. data levels missing from contrast: would generate empty/undefined rows.
    #    better to filter data frame first
    # 3. contrast levels missing from data: would have empty columns, generate a
    #    rank-deficient model matrix.
    c_levels = something(DataAPI.levels(contrasts), levels)
    if eltype(c_levels) != eltype(levels)
        throw(
            ArgumentError(
                "mismatching levels types: got $(eltype(levels)), expected " *
                "$(eltype(c_levels)) based on contrasts levels.",
            ),
        )
    end
    mismatched_levels = symdiff(c_levels, levels)
    if !isempty(mismatched_levels)
        throw(
            ArgumentError(
                "contrasts levels not found in data or vice-versa: " *
                "$mismatched_levels." *
                "\n  Data levels: $levels." *
                "\n  Contrast levels: $c_levels",
            ),
        )
    end

    n = length(c_levels)
    if n == 0
        throw(
            ArgumentError(
                "empty set of levels found (need at least two to compute " * "contrasts)."
            ),
        )
    elseif n == 1
        throw(
            ArgumentError(
                "only one level found: $(c_levels[1]) (need at least two to " *
                "compute contrasts).",
            ),
        )
    end

    # find index of base level. use contrasts.base, then default (1).
    base_level = baselevel(contrasts)
    baseind = base_level === nothing ? 1 : findfirst(isequal(base_level), c_levels)
    if baseind === nothing
        throw(ArgumentError("base level $(base_level) not found in levels " * "$c_levels."))
    end

    tnames = termnames(contrasts, c_levels, baseind)

    mat = contrasts_matrix(contrasts, baseind, n)

    return ContrastsMatrix(mat, tnames, c_levels, contrasts)
end

function ContrastsMatrix(c::Type{<:AbstractContrasts}, levels::AbstractVector)
    return throw(
        ArgumentError("contrast types must be instantiated (use $c() instead of $c)")
    )
end

# given an existing ContrastsMatrix, check that all passed levels are present
# in the contrasts. Note that this behavior is different from the
# ContrastsMatrix constructor, which requires that the levels be exactly the same.
function ContrastsMatrix(c::ContrastsMatrix, levels::AbstractVector)
    if !isempty(setdiff(levels, c.levels))
        throw(
            ArgumentError(
                "there are levels in data that are not in ContrastsMatrix: " *
                "$(setdiff(levels, c.levels))" *
                "\n  Data levels: $(levels)" *
                "\n  Contrast levels: $(c.levels)",
            ),
        )
    end
    return c
end

function termnames(C::AbstractContrasts, levels::AbstractVector, baseind::Integer)
    not_base = [1:(baseind - 1); (baseind + 1):length(levels)]
    return levels[not_base]
end

function Base.getindex(contrasts::ContrastsMatrix{C,T}, rowinds, colinds) where {C,T}
    return getindex(contrasts.matrix, getindex.(Ref(contrasts.invindex), rowinds), colinds)
end

mutable struct DummyCoding <: AbstractContrasts
    base::Any
    levels::Union{AbstractVector,Nothing}
end
## constructor with optional keyword arguments, defaulting to nothing
function DummyCoding(; base=nothing, levels::Union{AbstractVector,Nothing}=nothing)
    return DummyCoding(base, levels)
end
baselevel(c::DummyCoding) = c.base
DataAPI.levels(c::DummyCoding) = c.levels

# fallback method for other types that might not have base or level fields
baselevel(c::AbstractContrasts) = nothing
DataAPI.levels(c::AbstractContrasts) = nothing
