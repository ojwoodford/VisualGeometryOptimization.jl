using VisualGeometryOptimization, NLLSsolver, Static, StaticArrays 

@testset "warp.jl" begin
    # Test projective warp definition
    warp = HomographyWarp()
    @test NLLSsolver.nvars(warp) == static(8)

    # Test update function
    updatevec = [0.4, 0.1, 0.2, 0.01, 0.02, 0.03, -0.01, 0.04, 0.05, 0.7]
    updated_warp = NLLSsolver.update(warp, updatevec, 2)
    @test updated_warp.mat == SMatrix{3, 3, Float64, 9}(1.05, -0.02, 0.1,
                                                        0.0, 0.99, 0.2,
                                                        0.04, 0.05, 0.96)

    # Test transform function
    coords = SMatrix{2, 3, Float64, 6}(1.0, 2.0, 1.0, 3.0, 4.0, 1.0)
    gt = SMatrix{2, 3, Float64, 6}(0.7465753424657535, 1.3767123287671232, 0.6566265060240963, 1.8072289156626504, 2.717948717948718, 0.6153846153846154)
    transformed_coords = transform(updated_warp, coords)
    @test transformed_coords ≈ gt
    coords = SMatrix{3, 3, Float64, 9}(1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 0.5, 0.5)
    transformed_coords = transform(updated_warp, coords)
    @test transformed_coords ≈ gt
end