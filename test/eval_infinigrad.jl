using Pkg
Pkg.activate(@__DIR__)
using DrWatson, DataFrames, JLD2
using PixelFlipper
using Flux, Metalhead        # pre-trained vision models in Flux

## Load data
const input_data = load(datadir("input_batch.jld2")) # from save_input.jl
const input = input_data["input"]
const labels = input_data["labels"]

## Load & prepare model
const model = VGG(19; pretrain=true).layers

## Load computed results
df = collect_results(datadir("batched_heatmaps"))

## Define PixelFlipper
pf = PixelFlipping()

val = first(eachrow(df)).val    

res = run(pf, model, input, val)
