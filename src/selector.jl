"""
    AbstractSelector

Abstract supertype of all selectors. Given an `Explanation` or WHCN array, all `AbstractSelector` return an iterator of values to be imputed.
"""
abstract type AbstractSelector end

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

function select(sel::PixelSelector, x::AbstractArray{T,4}) where {T}
    x_reduced = reduce_color_channel(x, sel.reduce)
    I_reduced = sortinds(x_reduced)

    nchannels = size(x, 3)
    I = select_all_color_channels.(I_reduced, nchannels)
    return I
end

function select_all_color_channels(I::CartesianIndex, nchannels)
    iW, iH, _, iN = Tuple(I)
    return CartesianIndices((iW, iH, 1:nchannels, iN))
end

#=======#
# Utils #
#=======#

"""
    sortinds(A)

Return `CartesianIndices` of `A` sorted by decreasing value.
"""
function sortinds(A::AbstractArray{T,3}) where {T}
    idx_perm = sortperm(A[:]; rev=true)
    return CartesianIndices(A)[idx_perm]
end
function sortinds(A::AbstractArray{T,4}) where {T}
    transposed = [[row[i] for row in data] for i in 1:length(data[1])]
    return sortinds.(eachslice(A; dims=4))
end

function reduce_color_channel(val::AbstractArray{T,4}, method::Symbol) where {T}
    init = zero(eltype(val))
    if size(val, 3) == 1 # nothing to reduce
        return val
    elseif method == :sum
        return reduce(+, val; dims=3)
    elseif method == :maxabs
        return reduce((c...) -> maximum(abs.(c)), val; dims=3, init=init)
    elseif method == :norm
        return reduce((c...) -> sqrt(sum(c .^ 2)), val; dims=3, init=init)
    elseif method == :sumabs
        return reduce((c...) -> sum(abs, c), val; dims=3, init=init)
    elseif method == :abssum
        return reduce((c...) -> abs(sum(c)), val; dims=3, init=init)
    end
    throw( # else
        ArgumentError(
            "`reduce` :$method not supported, should be :maxabs, :sum, :norm, :sumabs, or :abssum",
        ),
    )
end
