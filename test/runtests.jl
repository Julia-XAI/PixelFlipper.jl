using PixelFlipper
using Test
using Aqua
using JET

@testset "PixelFlipper.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(PixelFlipper)
    end
    if VERSION < v"1.12" # TODO: remove. As of PR #4, JET fails on pre-release builds.
        @testset "Code linting (JET.jl)" begin
            JET.test_package(PixelFlipper; target_defined_modules=true)
        end
    end
    # Write your tests here.
    @testset "GPU tests" begin
        include("test_gpu.jl")
    end
end
