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
    pf = PixelFlipping(; steps=10, device=Array)
    @test_nowarn evaluate(pf, model, input, val)
end

@testset "Run pixel flipping (GPU)" begin
    pf = PixelFlipping(; steps=10, device=device)
    @testset "GPU explanation" begin
        @test_nowarn evaluate(pf, model_gpu, input_gpu, val_gpu)
    end
    # This should be avoided, but can happen when loading explanations from files
    @testset "CPU explanation" begin
        @test_nowarn evaluate(pf, model_gpu, input_gpu, val)
    end
end
