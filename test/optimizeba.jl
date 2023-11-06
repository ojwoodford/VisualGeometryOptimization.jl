using VisualGeometryOptimization, NLLSsolver, Test, StaticArrays, LinearAlgebra

function create_bal_problem(ncameras, nlandmarks, propvisible)
    problem = VisualGeometryOptimization.BALProblem()

    # Generate the cameras on a radius 10 sphere, pointing to the origin
    for i = 1:ncameras
        camcenter = normalize(randn(SVector{3, Float64}))
        addvariable!(problem, VisualGeometryOptimization.BALImage{Float64}(EffPose3D(Rotation3DL(SMatrix{3, 3}(0., 0., 1., 0., -1., 0., 1., 0., 0.) * UnitVec3D(camcenter).v.m'), Point3D(camcenter * 10)), SimpleCamera{Float64}(rand() * 10 + 1), BarrelDistortion{Float64}(randn()*1.e-3, randn()*1.e-5)))
    end
    
    # Generate the landmarks in a unit-sized cube centered on the origin
    for i = 1:nlandmarks
        addvariable!(problem, Point3D{Float64}(rand(SVector{3, Float64}) .- 0.5))
    end

    # Generate the measurements
    visibility = abs.(repeat(vec(1:ncameras), outer=(1, nlandmarks)) .- LinRange(2, ncameras-1, nlandmarks)')
    visibility = visibility .<= sort(vec(visibility))[Int(ceil(length(visibility)*propvisible))]
    for camind = 1:ncameras
        camera = problem.variables[camind]::VisualGeometryOptimization.BALImage{Float64}
        for (landmark, tf) in enumerate(view(visibility, camind, :)')
            if tf
                landmarkind = landmark + ncameras
                addcost!(problem, VisualGeometryOptimization.BALResidual(generatemeasurement(camera, problem.variables[landmarkind]::Point3D{Float64}), (camind, landmarkind)))
            end
        end
    end

    # Return the NLLSProblem
    return problem
end

function perturb_bal_problem(problem, pointnoise, posenoise)
    for ind in 1:lastindex(problem.variables)
        if isa(problem.variables[ind], Point3D)
            problem.variables[ind]::Point3D{Float64} = update(problem.variables[ind]::Point3D{Float64}, (randn(SVector{3, Float64})) * pointnoise)
        else
            problem.variables[ind]::VisualGeometryOptimization.BALImage{Float64} = update(problem.variables[ind]::VisualGeometryOptimization.BALImage{Float64}, vcat(randn(SVector{6, Float64}) * posenoise, zeros(SVector{3, Float64})))
        end
    end
    return problem
end

@testset "optimizeba.jl" begin
    # Generate some test data for a dense problem
    problem = create_bal_problem(3, 5, 1.0)
    
    # Optimze just the landmarks
    problem = perturb_bal_problem(problem, 0.1, 0.0)
    NLLSsolver.optimizesingles!(problem, NLLSOptions(), Point3D{Float64})
    @test cost(problem) < 1.e-10

    # Optimize problem
    problem = perturb_bal_problem(problem, 0.1, 0.01)
    result = optimize!(problem)
    @test cost(problem) == result.bestcost
    @test result.bestcost < 1.e-10

    # Generate & optimize a sparse problem
    problem = create_bal_problem(10, 50, 0.3)
    problem = perturb_bal_problem(problem, 0.3, 0.3)
    problem, result, aucs = optimizeBAproblem(problem; maxiters=20);
    @test result.bestcost < 1.e-10
    @test aucs[2] > aucs[1]
end
