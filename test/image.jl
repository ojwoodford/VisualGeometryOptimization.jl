using VisualGeometryOptimization, Test, TestImages, StaticArrays

@testset "image.jl" begin
    # Test pixel to image transformations
    halfimsz = SA[640.0, 480.0]
    x = SVector(7.0/11., 2.0/3.)
    xi = x .* (0.3 * halfimsz)
    imscale = ImageScale(halfimsz)
    y = pixel2image(imscale, xi)
    @test image2pixel(imscale, y) ≈ xi

    # Test warping of the weight matrix
    y_, W = pixel2image(imscale, xi, @SMatrix [1. 0.; 0. 1.])
    @test y_ == y
    @test W ≈ ForwardDiff.jacobian(x -> image2pixel(imscale, x), y)
    
end