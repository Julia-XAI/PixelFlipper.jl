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
@kwdef struct PixelSelector <: AbstractSelector
    reduce::Symbol = DEFAULT_REDUCE
end

# The type of index we are working with. Only required because I want to pre-allocate memory for it.
const PixelIndices = CartesianIndices{
    4,Tuple{Base.OneTo{Int64},Base.OneTo{Int64},UnitRange{Int64},Base.OneTo{Int64}}
}

"""
    select(x, selector)

Return matrix of `CartesianIndices` of `x` sorted by decreasing value.
Requires `x` to be in WHCN format, as each column in the output corresponds to an inputs in the batch.

## Example
```julia
julia> selector = PixelSelector()
PixelSelector(:norm)

julia> A = randn(2, 2, 1, 2)
2×2×1×2 Array{Float64, 4}:
[:, :, 1, 1] =
 -0.394689  -0.240333
  0.070783  -0.129964

[:, :, 1, 2] =
 0.858218  1.79357
 1.81366   0.152386

julia> PixelFlipper.select(A, selector)
4×2 Matrix{CartesianIndices{4, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}, UnitRange{Int64}, Base.OneTo{Int64}}}}:
 CartesianIndices((2, 1, 1:1, 1))  CartesianIndices((2, 1, 1:1, 2))
 CartesianIndices((2, 2, 1:1, 1))  CartesianIndices((1, 2, 1:1, 2))
 CartesianIndices((1, 2, 1:1, 1))  CartesianIndices((1, 1, 1:1, 2))
 CartesianIndices((1, 1, 1:1, 1))  CartesianIndices((2, 2, 1:1, 2))
```
"""
function select(x::AbstractWHCN, sel::PixelSelector)
    w, h, c, n = size(x)

    # Reduce color channel
    x_reduced = reduce_color_channel(x, sel.reduce)

    # Allocate output matrix of indices
    sorted_indices = Matrix{PixelIndices}(undef, w * h, n)

    # For each sample in batch, compute indices of sorted values 
    for (j, slice) in Iterators.enumerate(eachslice(x_reduced; dims=4))
        # Compute sorted vector of `CartesianIndex`es
        i_perm = sortperm(slice[:]; rev=true)
        Is = CartesianIndices(slice)[i_perm]

        # Rewrite each `CartesianIndex` into a `CartesianIndices` covering all color channels 
        for (i, I) in enumerate(Is)
            iw, ih, _ = Tuple(I) # unpack CartesianIndex
            sorted_indices[i, j] = CartesianIndices((iw, ih, 1:c, j))
        end
    end
    return sorted_indices
end
