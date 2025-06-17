using Test

@testset "VisualGeometryOptimization.jl" begin
    include("camera.jl")
    include("geometry.jl")
    include("image.jl")
    include("optimizeba.jl")
    include("normalize.jl")
    include("warp.jl")
end
