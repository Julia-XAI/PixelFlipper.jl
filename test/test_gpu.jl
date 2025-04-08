using PixelFlipper
using Test

using Flux
using MLUtils: flatten
using Metal, JLArrays

device = if Metal.functional()
    @info "Using Metal as GPU device"
    mtl # use Apple Metal locally
else
    @info "Using JLArrays as GPU device"
    jl # use JLArrays to fake GPU array
end

model = Chain(flatten, Dense(16 * 3 => 32, relu), Dense(32 => 5))
input = rand(Float32, 4, 4, 3, 8)
val = similar(input)
@test_nowarn model(input)

model_gpu = device(model)
input_gpu = device(input)
val_gpu = device(val)
@test_nowarn model_gpu(input_gpu)

@testset "Run pixel flipping (CPU)" begin
    pf = PixelFlipping(; steps=10, device=Array, show_progress=false)
    res = evaluate(pf, model, input, val)
    @test res isa PixelFlippingResult
end

@testset "Run pixel flipping (GPU)" begin
    pf = PixelFlipping(; steps=10, device=device, show_progress=false)
    @testset "GPU explanation" begin
        res = evaluate(pf, model_gpu, input_gpu, val_gpu)
        @test res isa PixelFlippingResult
    end
    # This should be avoided, but can happen when loading explanations from files
    @testset "CPU explanation" begin
        res = evaluate(pf, model_gpu, input_gpu, val)
        @test res isa PixelFlippingResult
    end
end
