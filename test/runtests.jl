using Test, Random
Random.seed!(1) # Fix the random test data, so test runs are reproducible

@testset "VisualGeometryOptimization.jl" begin
    include("utils.jl")
    include("camera.jl")
    include("geometry.jl")
    include("image.jl")
    include("optimizeba.jl")
    include("normalize.jl")
    include("warp.jl")
    include("essentialmatrixrefinement.jl")
end
