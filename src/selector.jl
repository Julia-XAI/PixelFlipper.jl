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

"""
    select(x, selector)

Return matrix of `CartesianIndices` of `x` sorted by decreasing value.
Requires `x` to be in WHCN format, as each column in the output corresponds to an inputs in the batch.

## Example
```julia
julia> selector = PixelSelector()
PixelSelector(:norm)

julia> A = randn(1, 2, 2, 2)
1×2×2×2 Array{Float64, 4}:
[:, :, 1, 1] =
 -1.9275  -3.01383

[:, :, 2, 1] =
 -0.424713  1.24167

[:, :, 1, 2] =
 -1.36198  1.21235

[:, :, 2, 2] =
 -1.75508  -0.700117

julia> PixelFlipper.select(A, selector)
2×4 Matrix{CartesianIndex{4}}:
 CartesianIndex(1, 2, 1, 1)  CartesianIndex(1, 2, 2, 1)  CartesianIndex(1, 1, 1, 2)  CartesianIndex(1, 1, 2, 2)
 CartesianIndex(1, 1, 1, 1)  CartesianIndex(1, 1, 2, 1)  CartesianIndex(1, 2, 1, 2)  CartesianIndex(1, 2, 2, 2)
```
"""
function select(x::AbstractWHCN, sel::PixelSelector)
    w, h, c, n = size(x)

    # Reduce color channel
    x_reduced = reduce_color_channel(x, sel.reduce)

    # Allocate output matrix of indices
    sorted_indices = Matrix{CartesianIndex{4}}(undef, w * h, c * n)

    # For each sample in batch, compute indices of sorted values 
    for (in, slice) in Iterators.enumerate(eachslice(x_reduced; dims=4))
        # Compute sorted vector of `CartesianIndex`es
        i_perm = sortperm(slice[:]; rev=true)
        Is = CartesianIndices(slice)[i_perm]

        # Rewrite each `CartesianIndex` into a `CartesianIndices` covering all color channels 
        for (i, I) in enumerate(Is)
            iw, ih, _ = Tuple(I) # unpack CartesianIndex
            for ic in 1:c
                sorted_indices[i, ic + c * (in - 1)] = CartesianIndex((iw, ih, ic, in))
            end
        end
    end
    return sorted_indices
end
