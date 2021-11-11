is_call(ex::Expr) = Meta.isexpr(ex, :call)
is_call(ex::Expr, op::Symbol) = Meta.isexpr(ex, :call) && ex.args[1] == op
is_call(::Any) = false
is_call(::Any, ::Any) = false
check_call(ex) = is_call(ex) || throw(ArgumentError("non-call expression encountered: $ex"))

function catch_dollar(ex::Expr)
    return Meta.isexpr(ex, :$) && throw(
        ArgumentError(
            "interpolation with \$ not supported in @formula.  Use @eval @formula(...) instead.",
        ),
    )
end

"""
    @formula(ex)
Capture and parse a formula expression as a `Formula` struct.

A formula is an abstract specification of a dependence between _left-hand_ and
_right-hand_ side variables as in, e.g., a regression model.  Each side
specifies at a high level how tabular data is to be converted to a numerical
matrix suitable for modeling.  This specification looks something like Julia
code, is represented as a Julia `Expr`, but uses special syntax.  The `@formula`
macro takes an expression like `y ~ a*b`, transforms it according to the
formula syntax rules into a lowered form (like `y ~ a + b + a&b`), and
constructs a `Formula` struct which captures the original expression, the
lowered expression, and the left- and right-hand-side.

Operators that have special interpretations in this syntax are

* `~` is the formula separator, where it is a binary operator (the first
  argument is the left-hand side, and the second is the right-hand side.

* `+` concatenates variables as columns when generating a model matrix.

* `&` representes an _interaction_ between two or more variables, which
  corresponds to a row-wise kronecker product of the individual terms
  (or element-wise product if all terms involved are continuous/scalar).

* `*` expands to all main effects and interactions: `a*b` is equivalent to
  `a+b+a&b`, `a*b*c` to `a+b+c+a&b+a&c+b&c+a&b&c`, etc.

The rules that are applied are

* The associative rule (un-nests nested calls to `+`, `&`, and `*`).
* The distributive rule (interactions `&` distribute over concatenation `+`).
* The `*` rule expands `a*b` to `a+b+a&b` (recursively).
* Single-argument `&` calls are stripped, so `&(x)` becomes the main effect `x`.
"""
macro formula(ex)
    is_call(ex, :~) || throw(ArgumentError("expected formula separator ~, got $(ex.head)"))
    length(ex.args) == 3 || throw(ArgumentError("malformed expression in formula $ex"))
    return terms!(sort_terms!(parse!(ex)))
end

"""
    abstract type FormulaRewrite end

Formula parsing is expressed as a bunch of expression re-writes, each of which
is a subtype of `FormulaRewrite`.  There are two methods that dispatch on these
types: `applies(ex, child_idx, rule::Type{<:FormulaRewrite})` checks whether the
re-write `rule` needs to be applied at argument `child_idx` of expression `ex`,
and `rewrite!(ex, child_idx, rule::Type{<:FormulaRewrite})` re-writes `ex`
according to `rule` at position `child_idx`, and returns the next `child_idx`
that needs to be checked.
"""
abstract type FormulaRewrite end

"""
    struct Star <: FormulaRewrite end

Expand `a*b` to `a + b + a&b` (`*(a,b)` to `+(a,b,&(a,b))`).  Applies
recursively to multiple `*` arguments, so needs a clean-up pass (from
distributive/associative).
"""
struct Star <: FormulaRewrite end
applies(ex::Expr, child_idx::Int, ::Type{Star}) = is_call(ex.args[child_idx], :*)
expand_star(a, b) = Expr(:call, :+, a, b, Expr(:call, :&, a, b))
function rewrite!(ex::Expr, child_idx::Int, ::Type{Star})
    child = ex.args[child_idx]
    @debug "  expand star: $ex -> "
    child.args = reduce(expand_star, child.args[2:end]).args
    @debug "               $ex"
    return child_idx
end

"""
    struct AssociativeRule <: FormulaRewrite end

Apply associative rule: if in an expression headed by an associative operator
(`+,&,*`) and the sub-expression `child_idx` is headed by the same operator,
splice that child's children into it's location.
"""
struct AssociativeRule <: FormulaRewrite end
const ASSOCIATIVE = (:+, :&, :*)
function applies(ex::Expr, child_idx::Int, ::Type{AssociativeRule})
    return is_call(ex) &&
           ex.args[1] in ASSOCIATIVE &&
           is_call(ex.args[child_idx], ex.args[1])
end
function rewrite!(ex::Expr, child_idx::Int, ::Type{AssociativeRule})
    @debug "    associative: $ex -> "
    splice!(ex.args, child_idx, ex.args[child_idx].args[2:end])
    @debug "                 $ex"
    return child_idx
end

"""
    struct Distributive <: FormulaRewrite end

Distributive propery: `&(a..., +(b...), c...)` to `+(&(a..., b_i, c...)_i...)`.
Replace outer call (:&) with inner call (:+), whose arguments are copies of the
outer call, one for each argument of the inner call.  For the ith new child, the
original inner call is replaced with the ith argument of the inner call.
"""
struct Distributive <: FormulaRewrite end
const DISTRIBUTIVE = Set([:& => :+])
function applies(ex::Expr, child_idx::Int, ::Type{Distributive})
    return is_call(ex) &&
           is_call(ex.args[child_idx]) &&
           (ex.args[1] => ex.args[child_idx].args[1]) in DISTRIBUTIVE
end
function rewrite!(ex::Expr, child_idx::Int, ::Type{Distributive})
    @debug "    distributive: $ex -> "
    new_args = deepcopy(ex.args[child_idx].args)
    for i in 2:length(new_args)
        new_child = deepcopy(ex)
        new_child.args[child_idx] = new_args[i]
        new_args[i] = new_child
    end
    ex.args = new_args
    @debug "                  $ex"
    return 2
end

"""
    And1 <: FormulaRewrite

Remove numbers from interaction terms, so `1&x` becomes `&(x)` (which is later
cleaned up by `EmptyAnd`).
"""
struct And1 <: FormulaRewrite end
function applies(ex::Expr, child_idx::Int, ::Type{And1})
    return is_call(ex, :&) && ex.args[child_idx] isa Number
end
function rewrite!(ex::Expr, child_idx::Int, ::Type{And1})
    @debug "    &1: $ex ->"
    ex.args[child_idx] == 1 ||
        @warn "Number $(ex.args[child_idx]) removed from interaction term $ex"
    deleteat!(ex.args, child_idx)
    @debug "        $ex"
    return child_idx
end

# default re-write is a no-op (go to next child)
rewrite!(ex::Expr, child_idx::Int, ::Nothing) = child_idx + 1

# like `findfirst` but returns the first element where predicate is true, or
# nothing
function filterfirst(f::Function, a::AbstractArray)
    idx = findfirst(f, a)
    return idx === nothing ? nothing : a[idx]
end

const SPECIALS = (:+, :&, :*, :~)

parse!(x) = parse!(x, [And1, Star, AssociativeRule, Distributive])
parse!(x, rewrites) = x
function parse!(ex::Expr, rewrites::Vector)
    @debug "parsing $ex"
    catch_dollar(ex)
    check_call(ex)

    # parse a copy of non-special calls
    ex_parsed = ex.args[1] ∉ SPECIALS ? deepcopy(ex) : ex

    # iterate over children, checking for special rules
    child_idx = 2
    while child_idx <= length(ex_parsed.args)
        @debug "  ($(ex_parsed.args[1])) i=$child_idx: $(ex_parsed.args[child_idx])"
        # depth first: parse each child first
        parse!(ex_parsed.args[child_idx], rewrites)
        # find first rewrite rule that applies
        rule = filterfirst(r -> applies(ex_parsed, child_idx, r), rewrites)
        # re-write according to that rule and update the child to position rewrite nex_parsedt
        child_idx = rewrite!(ex_parsed, child_idx, rule)
    end
    @debug "done: $ex_parsed"

    if ex.args[1] ∈ SPECIALS
        return ex_parsed
    end
end

# generate Term expressions for symbols
terms!(::Nothing) = :(nothing)
terms!(s::Symbol) = :(Term($(Meta.quot(s))))
function terms!(ex::Expr)
    if ex.args[1] ∈ SPECIALS
        ex.args[1] = esc(ex.args[1])
        ex.args[2:end] .= terms!.(ex.args[2:end])
    end
    return ex
end

function sort_terms!(ex::Expr)
    check_call(ex)
    if ex.args[1] ∈ ASSOCIATIVE
        sort!(view(ex.args, 2:length(ex.args)); by=degree)
    else
        # recursively sort children
        sort_terms!.(ex.args)
    end
    return ex
end
sort_terms!(x) = x

degree(i::Integer) = 0
degree(::Symbol) = 1
function degree(ex::Expr)
    check_call(ex)
    if ex.args[1] == :&
        length(ex.args) - 1
    elseif ex.args[1] == :|
        # put ranef terms at end
        typemax(Int)
    else
        # arbitrary functions are treated as main effect terms
        1
    end
end
