using VisualGeometryOptimization, Random, StaticArrays, ForwardDiff, BenchmarkTools

function test_normalize(vec)
    result = vec .- (sum(vec) / length(vec))
    result = result ./ sqrt(result' * result)
    vec = mean0norm1!!(vec)
    @test result ≈ vec
end

@testset "normalize.jl" begin
    # Normal vectors
    A = randn(16)
    test_normalize(A)
    A = randn(1024)
    test_normalize(A)

    # Static vectors
    B = SVector{16, Float64}(randn(16))
    test_normalize(B)

    # Dual numbers


end