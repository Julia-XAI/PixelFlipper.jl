module PixelFlipper

using XAIBase
using NNlib: softmax
using Base: @kwdef
using ProgressMeter: @showprogress
using Statistics: mean
using UnicodePlots: lineplot, lineplot!

const AbstractWHCN{T} = AbstractArray{T,4}

include("utils.jl")
include("selector.jl")
include("imputer.jl")

const DEFAULT_SELECTOR = PixelSelector()
const DEFAULT_IMPUTER = ConstantImputer()
const DEFAULT_STEPS = 25
const DEFAULT_OUTPUT_SELECTOR = MaxActivationSelector()

"""
    PixelFlipping([; selector, imputer, steps])

Computes pixel flipping curves.

## Keyword arguments
- `selector::AbstractSelector`: Specify input selector. Defaults to `$DEFAULT_SELECTOR`.
- `imputer::AbstractImputer`: Specify input imputer. Defaults to `$DEFAULT_IMPUTER` of value zero.
- `steps::Int`: Specify number of imputation steps. Has to be smaller than the amount of selectable inputs in a sample. Defaults to `25`.
"""
@kwdef struct PixelFlipping{
    S<:AbstractSelector,I<:AbstractImputer,O<:AbstractOutputSelector
}
    selector::S = DEFAULT_SELECTOR
    imputer::I = DEFAULT_IMPUTER
    output_selector::O = DEFAULT_OUTPUT_SELECTOR
    steps::Int = 20
end

include("result.jl")
include("eval.jl")

export PixelFlipping, evaluate
export PixelFlippingResult
export ConstantImputer, PixelSelector
export mif, lif, srg, steps
export unicode_plot

end # module
