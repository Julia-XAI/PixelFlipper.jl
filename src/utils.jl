# Reduce color channels with an `AbstractPooling`,
# so that the metric evaluates the same reduction that is visualized in the heatmap.
# The pooling comes either from the evaluated `Attribution`
# or from the `PixelSelector`'s `reduce` (see `select` and `evaluate`).
function reduce_color_channel(val::AbstractArray{T, 4}, pooling::AbstractPooling) where {T}
    size(val, 3) == 1 && return val # nothing to reduce
    reduced = pool(pooling, val, 3) # reduces over color channels, dropping that dimension
    w, h, n = size(reduced)
    return reshape(reduced, w, h, 1, n)
end
