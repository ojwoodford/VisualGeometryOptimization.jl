using VisualGeometryOptimization, Test

@testset "utils.jl" begin
    # Test computeauc against hand-computed areas under the piecewise-linear error-recall curve
    @test computeauc([0.5, 0.8], 1.0) ≈ 0.55 # All errors below the threshold
    @test computeauc([0.5, 3.0], 1.0) ≈ 0.4 # Some errors above the threshold
    @test computeauc([-0.5, 3.0], 1.0) ≈ 0.4 # Signs are ignored
    @test computeauc([1.0, 6.0], 2.0) ≈ 0.4 # Errors are scaled by the threshold
    @test computeauc([3.0, 4.0], 1.0) ≈ 1 / 12 # All errors above the threshold
    @test computeauc(zeros(10), 1.0) ≈ 1.0 # Perfect
    # The AUC increases as errors decrease
    errors = rand(100) * 3
    @test computeauc(errors * 0.5, 1.0) > computeauc(errors, 1.0)
end
