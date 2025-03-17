module PixelFlipper

using XAIBase
using ImageCore
using ColorTypes
using Base: @kwdef

include("utils.jl")
include("selector.jl")
include("imputer.jl")

"""
    PixelFlipping(analyzer::AbstractXAIMethod, model, input)

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
struct PixelFlipping{A,M,I<:AbstractImputer}
    analyzer::A
    model::M
    imputer::I
    steps::Int
end

struct Result{T}
    mif::Vector{T}
    lif::Vector{T}
end


export PixelFlipping, Result, ConstantImputer, PixelSelector

end # module
