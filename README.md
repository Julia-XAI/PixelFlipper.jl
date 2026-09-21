# PixelFlipper.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Julia-XAI.github.io/PixelFlipper.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Julia-XAI.github.io/PixelFlipper.jl/dev/) -->
[![Build Status](https://github.com/Julia-XAI/PixelFlipper.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Julia-XAI/PixelFlipper.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Julia-XAI/PixelFlipper.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Julia-XAI/PixelFlipper.jl)
[![Code Style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

Julia implementation of pixel flipping to evaluate XAI methods.

Pixel flipping measures how faithful a feature attribution is to the model it explains.
This package implements the [symmetric relevance gain (SRG)][srg-paper] metric.

## Installation

This package supports Julia ≥1.10. To install it, open the Julia REPL and run
```julia-repl
julia> ]add PixelFlipper
```

## Example

PixelFlipper.jl evaluates attributions from any method in the [Julia-XAI ecosystem][juliaxai-docs].
Here we score an `InputTimesGradient` attribution from
[ExplainableAI.jl](https://github.com/Julia-XAI/ExplainableAI.jl):

```julia
using PixelFlipper
using ExplainableAI          # XAI methods that produce attributions
using Zygote                 # autodiff backend for gradient-based methods
using Flux, Metalhead        # pre-trained vision models in Flux

# Load & prepare model
model = VGG(16, pretrain=true).layers

# Load a batch of preprocessed inputs in WHCN format
# (width, height, color channels, batch), e.g. using DataAugmentation.jl
input = ...

# Pick an XAI method to evaluate
analyzer = InputTimesGradient(model)

# Run pixel flipping
pf = PixelFlipping(steps=100)
result = evaluate(pf, model, input, analyzer)
```

The `evaluate` call runs the analyzer for you.
If you already have an attribution or a raw WHCN array of relevance values,
you can pass it directly instead of the analyzer:

```julia
attr = analyze(input, analyzer)   # an XAIBase `Attribution`
result = evaluate(pf, model, input, attr)
```

Inspect the resulting MIF and LIF curves and their SRG score:

```julia
unicode_plot(result)   # plot both curves in the terminal
srg(result)            # symmetric relevance gain score
mif(result)            # most influential first curve
lif(result)            # least influential first curve
```

`PixelFlipping` is configurable through its keyword arguments:
* `selector` chooses how relevance is reduced over color channels and sorted
  when you pass a raw WHCN array
  (an `Attribution` is reduced with its own `pooling` instead),
* `imputer` sets the value that occluded features are replaced with,
* `steps` sets the number of occlusion steps
  (it has to be smaller than the number of selectable features per sample),
* `output_selector` picks which model output each sample is scored against,
  defaulting to the maximally activated neuron,
* `show_progress` toggles the progress meter, and
* `device` selects the array type, e.g. CUDA.jl's `cu` for GPU support.

## Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry of Education and Research under the grant BIFOLD26B.

[juliaxai-docs]: https://julia-xai.github.io/XAIDocs/
[srg-paper]: https://arxiv.org/abs/2401.06654
