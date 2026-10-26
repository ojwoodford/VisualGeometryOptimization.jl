using Test, Random
Random.seed!(1) # Fix the random test data, so test runs are reproducible
@testset "VisualGeometryOptimization.jl" begin
    include("utils.jl")
    include("camera.jl")
    include("geometry.jl")
    include("optimizeba.jl")
    include("essentialmatrixrefinement.jl")
end
