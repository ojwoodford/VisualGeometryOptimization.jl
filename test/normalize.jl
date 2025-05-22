using VisualGeometryOptimization, Test, Random, StaticArrays

myapprox(A, B) = A ≈ B
function myapprox(A::T, B::T) where T <: AbstractVector{FD} where FD <: ForwardDiff.Dual
    for i in eachindex(A)
        if !(ForwardDiff.value(A[i]) ≈ ForwardDiff.value(B[i]))
            return false
        end
        if !(ForwardDiff.partials(A[i]) ≈ ForwardDiff.partials(B[i]))
            return false
        end
    end
    return true
end

function test_normalize(vec)
    result = vec .- (sum(vec) / length(vec))
    result = result ./ sqrt(result' * result)
    vec = mean0norm1!!(vec)
    return myapprox(result, vec)
end

@testset "normalize.jl" begin
    # Normal vectors
    A = randn(16)
    @test test_normalize(A)
    A = randn(1024)
    @test test_normalize(A)

    # Static vectors
    B = SVector{16, Float64}(randn(16))
    @test test_normalize(B)
end