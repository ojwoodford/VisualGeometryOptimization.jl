using VisualGeometryOptimization, NLLSsolver, Static, StaticArrays 

@testset "warp.jl" begin
    # Test projective warp definition
    warp = VisualGeometryOptimization.HomographyWarp()
    @test NLLSsolver.nvars(warp) == static(8)

    # Test update function
    updatevec = [0.4, 0.1, 0.2, 0.01, 0.02, 0.03, -0.01, 0.04, 0.05, 0.7]
    updated_warp = NLLSsolver.update(warp, updatevec, 2)
    @test updated_warp.mat == SMatrix{3, 3, Float64, 9}(1.05, -0.02, 0.1,
                                                        0.0, 0.99, 0.2,
                                                        0.04, 0.05, 0.96)

    # # Test transform function
    # coords = SMatrix{2, 3, Float64, 6}(1.0, 2.0, 1.0, 3.0, 4.0, 1.0)
    # transformed_coords = transform(warp, coords)
    # @test transformed_coords ≈ coords
end