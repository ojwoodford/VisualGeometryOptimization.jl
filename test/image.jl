using VisualGeometryOptimization, Test, TestImages, StaticArrays

@testset "image.jl" begin
    # Test pixel to image transformations
    halfimsz = SA[640.0, 480.0]
    x = SVector(7.0/11., 2.0/3.) .* (0.3 * halfimsz)
    imscale = ImageScale(halfimsz)
    y = pixel2image(imscale, x)
    @test image2pixel(imscale, y) ≈ x
    # Test warping of the weight matrix
    y_, W = pixel2image(imscale, x, @SMatrix [1. 0.; 0. 1.])
    @test y_ == y
    @test W ≈ ForwardDiff.jacobian(x -> image2pixel(imscale, x), y)

    # Test image pyramid generation
    impyr = ImagePyramid(Image(testimage("cameraman.tif"), imscale))
    @test length(impyr.level) == 5
    @test size(impyr.level[1]) == (512, 512)
    @test size(impyr.level[5]) == (32, 32)

    # Test sampling from the image
    
end