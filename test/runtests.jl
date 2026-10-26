using Test
@testset "VisualGeometryOptimization.jl" begin
    include("utils.jl")
    include("camera.jl")
    include("geometry.jl")
    include("optimizeba.jl")
    include("essentialmatrixrefinement.jl")
end
