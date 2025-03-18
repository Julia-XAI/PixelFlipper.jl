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
