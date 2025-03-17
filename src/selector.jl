"""
    AbstractSelector

Abstract supertype of all selectors. Given an `Explanation`, all `AbstractSelector` return an iterator of values to be imputed.

Currently, only the selection of pixels is supported.
In the future, this could be extended to subpixels or a superpixels.
"""
abstract type AbstractSelector end
abstract type AbstractPixelSelector <: AbstractSelector end

# Nice to have in the future:
# abstract type AbstractSubPixelSelector <: AbstractSelector end
# abstract type AbstractSuperPixelSelector <: AbstractSelector end

const DEFAULT_REDUCE = :norm

"""
    PixelSelector()
    PixelSelector(; reduce=:norm)

Reduces color channels in an `Explanation` according to `reduce` and returns an iterator over the indices of sorted values.

## Keyword arguments
- `reduce::Symbol`: Selects how color channels are reduced to a single number to apply a color scheme.
  The following methods can be selected, which are then applied over the color channels
  for each "pixel" in the array:
  - `:sum`: sum up color channels
  - `:norm`: compute 2-norm over the color channels
  - `:maxabs`: compute `maximum(abs, x)` over the color channels
  - `:sumabs`: compute `sum(abs, x)` over the color channels
  - `:abssum`: compute `abs(sum(x))` over the color channels
  Defaults to `:$DEFAULT_REDUCE`.
"""
@kwdef struct PixelSelector
    reduce::Symbol = DEFAULT_REDUCE
end

function select(sel::PixelSelector, expl::Explanation)
    vals = reduce_color_channel(expl.val, sel.reduce)
    return sortinds(vals)
end

#=======#
# Utils #
#=======#

"""
    sortinds(A)

Return `CartesianIndices` of `A` sorted by decreasing value.
"""
function sortinds(A::AbstractArray)
    I = sortperm(A[:]; rev = true)
    return CartesianIndices(A)[I]
end

function reduce_color_channel(val::AbstractArray, method::Symbol)
    init = zero(eltype(val))
    if size(val, 3) == 1 # nothing to reduce
        return val
    elseif method == :sum
        return reduce(+, val; dims = 3)
    elseif method == :maxabs
        return reduce((c...) -> maximum(abs.(c)), val; dims = 3, init = init)
    elseif method == :norm
        return reduce((c...) -> sqrt(sum(c .^ 2)), val; dims = 3, init = init)
    elseif method == :sumabs
        return reduce((c...) -> sum(abs, c), val; dims = 3, init = init)
    elseif method == :abssum
        return reduce((c...) -> abs(sum(c)), val; dims = 3, init = init)
    end
    throw( # else
        ArgumentError(
            "`reduce` :$method not supported, should be :maxabs, :sum, :norm, :sumabs, or :abssum",
        ),
    )
end
