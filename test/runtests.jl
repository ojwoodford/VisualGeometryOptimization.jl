using Test
@testset "VisualGeometryOptimization.jl" begin
    include("camera.jl")
    include("geometry.jl")
    include("optimizeba.jl")
end
