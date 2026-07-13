using PixelFlipper
using Test

@testset "PixelFlipper.jl" begin
    @testset verbose = true "Linting" begin
        @info "Running linting tests..."
        include("linting.jl")
    end

    @testset "GPU tests" begin
        include("test_gpu.jl")
    end
end
