using VisualGeometryOptimization, Random, StaticArrays, ForwardDiff, BenchmarkTools

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
    @test myapprox(result, vec)
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
    C = Vector{ForwardDiff.Dual{Float64, Float64, 2}}(undef, 16)
    for i in eachindex(C)
        C[i] = ForwardDiff.Dual{Float64, Float64, 2}(randn(), ForwardDiff.Partials{2, Float64}((randn(), randn())))
    end
    test_normalize(C)
end