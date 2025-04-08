# using PixelFlipping
using Test

using Flux
using Metal, JLArrays

if Metal.functional()
    @info "Using Metal as GPU device"
    device = mtl # use Apple Metal locally
else
    @info "Using JLArrays as GPU device"
    device = jl # use JLArrays to fake GPU array
end

model = Chain(Dense(10 => 32, relu), Dense(32 => 5))
input = rand(Float32, 10, 8)
val = similar(input)
@test_nowarn model(input)

model_gpu = device(model)
input_gpu = device(input)
val_gpu = device(expl)
@test_nowarn model_gpu(input_gpu)

pf = PixelFlipping(; steps=10)

@testset "Run pixel flipping (CPU)" begin
    evaluate(pf, model, input, val)
end

@testset "Run pixel flipping (GPU)" begin
    @testset "GPU explanation" begin
        evaluate(pf, model_gpu, input_gpu, val_gpu)
    end
    # This should be avoided, but can happen when loading explanations from files
    @testset "CPU explanation" begin
        evaluate(pf, model_gpu, input_gpu, val)
    end
end
