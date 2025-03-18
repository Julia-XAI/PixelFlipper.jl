module PixelFlipper

using XAIBase
using ImageCore
using ColorTypes
using Base: @kwdef

include("utils.jl")
include("selector.jl")
include("imputer.jl")

const DEFAULT_SELECTOR = PixelSelector()
const DEFAULT_IMPUTER = ConstantImputer()

"""
    PixelFlipping(model, input, explanation::Explanation)
    PixelFlipping(model, input, values::AbstractArray) 
    PixelFlipping(model, input, analyzer::AbstractXAIMethod)

Compute pixel flipping curves for given model, input batch and explanation.

## Keyword arguments
- `selector::AbstractSelector`: Specify input selector. Defaults to `PixelSelector(; reduce=:norm)` 
- `imputer::AbstractImputer`: Specify input imputer. Defaults to `ConstantImputer()` of value zero.
"""
@kwdef struct PixelFlipping{
    M,AI<:AbstractArray{T,4},AV<:AbstractArray{T,4},S<:AbstractSelector,I<:AbstractImputer
} where {T}
    model::M
    input::AI
    values::AV
    selector::S
    imputer::I
    steps::Int
end

function PixelFlipping(model, input, expl::Explanation; kwargs...)
    return PixelFlipping(model, input, expl.val; kwargs...)
end

function PixelFlipping(model, input, analyzer::AbstractXAIMethod; kwargs...)
    expl = analyze(input, analyzer)
    return PixelFlipping(model, input, expl; kwargs...)
end

struct Result{T}
    mif::Vector{T}
    lif::Vector{T}
end

export PixelFlipping, Result, ConstantImputer, PixelSelector

end # module
