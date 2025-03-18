module PixelFlipper

using XAIBase
using ImageCore
using ColorTypes
using Base: @kwdef

const AbstractWHCN{T} = AbstractArray{T,4}

include("utils.jl")
include("selector.jl")
include("imputer.jl")

const DEFAULT_SELECTOR = PixelSelector()
const DEFAULT_IMPUTER = ConstantImputer()
const DEFAULT_STEPS = 25

"""
    PixelFlipping([; selector, imputer, steps])

Computes pixel flipping curves.

## Keyword arguments
- `selector::AbstractSelector`: Specify input selector. Defaults to `$DEFAULT_SELECTOR`.
- `imputer::AbstractImputer`: Specify input imputer. Defaults to `$DEFAULT_IMPUTER` of value zero.
- `steps::Int`: Specify number of imputation steps. Has to be smaller than the amount of selectable inputs in a sample. Defaults to `25`.
"""
@kwdef struct PixelFlipping{S<:AbstractSelector,I<:AbstractImputer}
    selector::S = DEFAULT_SELECTOR
    imputer::I = DEFAULT_IMPUTER
    steps::Int = 25
end

"""
    run(pixelflipping, model, input, explanation::Explanation)
    run(pixelflipping, model, input, values::AbstractArray) 
    run(pixelflipping, model, input, analyzer::AbstractXAIMethod)

Run the `PixelFlipping` method on the given model, input and explanation.
"""
function run(pf::PixelFlipping, model, input::AbstractWHCN, val::AbstractWHCN)
    selection = select(pf.selector, val)

    return nothing
end

function run(pf::PixelFlipping, model, input::AbstractWHCN, expl::Explanation)
    return run(pf, model, input, expl.val)
end

function run(pf::PixelFlipping, model, input::AbstractWHCN, analyzer::AbstractXAIMethod;)
    expl = analyze(input, analyzer)
    return run(pf, model, input, expl)
end

struct PixelFlippingResult{T}
    mif::Vector{T}
    lif::Vector{T}
end

export PixelFlipping, Result, ConstantImputer, PixelSelector

end # module
