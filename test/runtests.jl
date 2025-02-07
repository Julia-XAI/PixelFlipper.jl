using PixelFlipper
using Test
using Aqua
using JET

@testset "PixelFlipper.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(PixelFlipper)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(PixelFlipper; target_defined_modules = true)
    end
    # Write your tests here.
end
