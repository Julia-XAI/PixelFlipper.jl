"""
    AbstractSelector

Abstract supertype of all selectors. Given an `Attribution` or WHCN array, all `AbstractSelector` return an iterator of values to be imputed.
"""
abstract type AbstractSelector end

const DEFAULT_REDUCE = NormPooling()

"""
    PixelSelector()
    PixelSelector(; reduce=NormPooling())

Reduces color channels in an `Attribution` according to `reduce` and returns an iterator over the indices of sorted values.

## Keyword arguments
- `reduce::AbstractPooling`: XAIBase pooling used to reduce color channels to a single number per "pixel" before sorting,
  e.g. `SumPooling`, `NormPooling`, `MaxAbsPooling`, `SumAbsPooling` or `AbsSumPooling`.
  Defaults to `$DEFAULT_REDUCE`.
  This is only used for raw WHCN arrays:
  an `Attribution` reduces channels with its own `pooling` instead (see `evaluate`).
"""
@kwdef struct PixelSelector{P <: AbstractPooling} <: AbstractSelector
    reduce::P = DEFAULT_REDUCE
end

"""
    select(x, selector)

Return matrix of `CartesianIndices` of `x` sorted by decreasing value.
Requires `x` to be in WHCN format, as each column in the output corresponds to an inputs in the batch.

## Example
```julia
julia> selector = PixelSelector()
PixelSelector{NormPooling}(NormPooling())

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
select(x::AbstractWHCN, sel::PixelSelector) = select(x, sel.reduce)

# `reduction` is an `AbstractPooling`,
# either a `PixelSelector`'s `reduce` or the pooling carried by an `Attribution` (see `evaluate`).
function select(x::AbstractWHCN, reduction)
    w, h, c, n = size(x)

    # Reduce color channel
    x_reduced = reduce_color_channel(x, reduction)

    # Allocate output matrix of indices
    sorted_indices = Matrix{CartesianIndex{4}}(undef, w * h, c * n)

    # For each sample in batch, compute indices of sorted values
    for (in, slice) in Iterators.enumerate(eachslice(x_reduced; dims = 4))
        # Compute sorted vector of `CartesianIndex`es
        i_perm = sortperm(slice[:]; rev = true)
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
