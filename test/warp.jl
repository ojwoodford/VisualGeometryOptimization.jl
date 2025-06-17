using VisualGeometryOptimization, NLLSsolver, Static 

@testset "warp.jl" begin
    # Test projective warp definition
    warp = VisualGeometryOptimization.HomographyWarp()
    @test NLLSsolver.nvars(warp) == static(8)

    # # Test update function
    # updatevec = zeros(7)
    # updated_warp = NLLSsolver.update(warp, updatevec)
    # @test updated_warp.mat == warp.mat

    # # Test transform function
    # coords = SMatrix{2, 3, Float64, 6}(1.0, 2.0, 1.0, 3.0, 4.0, 1.0)
    # transformed_coords = transform(warp, coords)
    # @test transformed_coords ≈ coords
end