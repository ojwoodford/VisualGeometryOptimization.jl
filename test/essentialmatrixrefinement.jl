using VisualGeometryOptimization, VisualGeometryDatasets, NLLSsolver, Test, StaticArrays, LinearAlgebra, Static
import VisualGeometryOptimization: PairwisePose, EpipolarResidualPair, EssentialMatrixProblem, getvec

# Calibration matrix with a single focal length and a principal point
calibration(f, cx, cy) = SMatrix{3, 3, Float64, 9}(f, 0., 0., 0., f, 0., cx, cy, 1.)

# Generate a synthetic two-view pair (no dataset download needed). Points are in front of both cameras,
# and X_B = R * X_A + t. Pixel noise of standard deviation `noise` is added to the image B points.
function create_twoview_pair(npoints, R, t, KA, KB; noise=0.0)
    x1 = SVector{2, Float64}[]
    x2 = SVector{2, Float64}[]
    while length(x1) < npoints
        XA = SVector(rand() - 0.5, rand() - 0.5, 0.0) * 4 + SVector(0., 0., 5 + 5 * rand())
        XB = R * XA + t
        XB[3] > 0.1 || continue
        push!(x1, project(KA * XA))
        push!(x2, project(KB * XB) + randn(SVector{2, Float64}) * noise)
    end
    return TwoViewPair{Float64}("synthetic", x1, x2, zeros(npoints), KA, KB, nothing, nothing, SMatrix{3, 3, Float64, 9}(R), SVector{3, Float64}(t), "", "")
end

# Cross product matrix: skew(a) * b = a × b
skew(a) = SMatrix{3, 3, Float64, 9}(0., a[3], -a[2], -a[3], 0., a[1], a[2], -a[1], 0.)

# Pixel distance of point x (in pixels) from homogeneous line l (in pixels)
linedistance(l, x) = abs(dot(l, push(x, 1))) / norm(pop(l))

# Rotation and unit-translation errors (in radians) between a pose and the ground truth
function poseerrors(pose::PairwisePose, R, t)
    Rerr = acos(clamp((tr(pose.poseAtoB.rot.m' * R) - 1) / 2, -1, 1))
    terr = acos(clamp(abs(dot(getvec(pose.poseAtoB.trans), normalize(t))), -1, 1))
    return Rerr, terr
end

@testset "essentialmatrixrefinement.jl" begin
    R = rodrigues(0.1, -0.2, 0.05)
    t = SVector(1.0, 0.2, 0.3)
    KA, KB = calibration(800., 320., 240.), calibration(1200., 500., 380.)

    # Test the pose variable
    pose = PairwisePose(R, t, KA[1,1], KB[1,1])
    @test pose.poseAtoB.rot.m ≈ R
    @test getvec(pose.poseAtoB.trans) ≈ normalize(t)
    @test nvars(pose) == 5
    pose2 = update(pose, zeros(5))
    @test pose2.poseAtoB.rot.m ≈ R && getvec(pose2.poseAtoB.trans) ≈ normalize(t)
    pose2 = update(pose, SVector(0.01, -0.02, 0.03, 0.01, -0.01))
    @test isa(pose2, PairwisePose{Float64})
    @test !(pose2.poseAtoB.rot.m ≈ R) && !(getvec(pose2.poseAtoB.trans) ≈ normalize(t))
    @test norm(getvec(pose2.poseAtoB.trans)) ≈ 1

    # Test problem construction
    data = create_twoview_pair(50, R, t, KA, KB)
    problem = makeEssentialMatrixproblem(data)
    @test isa(problem, EssentialMatrixProblem)
    @test length(problem.variables) == 1
    residuals = problem.costs.data[EpipolarResidualPair]
    @test length(residuals) == 50
    # Points are stored in metric coordinates
    @test residuals[1].xA ≈ project(KA \ push(data.x1[1], 1))
    @test residuals[1].xB ≈ project(KB \ push(data.x2[1], 1))
    # Unsupported calibration matrices are rejected
    badK = SMatrix{3, 3, Float64, 9}(800., 0., 0., 1., 800., 0., 320., 240., 1.) # Non-zero skew
    @test_throws AssertionError makeEssentialMatrixproblem(create_twoview_pair(5, R, t, badK, KB))
    badK = calibration(800., 320., 240.) .* SVector(1., 1.1, 1.) # Different x and y focal lengths
    @test_throws AssertionError makeEssentialMatrixproblem(create_twoview_pair(5, R, t, badK, KB))

    # Test the residual
    res = residuals[1]
    @test ndeps(res) == 1
    @test nres(res) == 2
    @test eltype(res) == Float64
    @test getvars(res, problem.variables) == (pose,)
    # Exact correspondences have zero error
    @test all(r -> maximum(abs, computeresidual(r, pose)) < 1.e-9, residuals)
    # The errors equal the pixel distances from the epipolar lines, computed using the fundamental matrix
    data = create_twoview_pair(50, R, t, KA, KB; noise=2.0)
    problem = makeEssentialMatrixproblem(data)
    F = inv(KB)' * skew(normalize(t)) * R * inv(KA)
    for (r, xA, xB) in zip(problem.costs.data[EpipolarResidualPair], data.x1, data.x2)
        err = computeresidual(r, pose)
        @test abs(err[1]) ≈ linedistance(F * push(xA, 1), xB)
        @test abs(err[2]) ≈ linedistance(F' * push(xB, 1), xA)
    end
    # The errors are finite when the translation is parallel to the image plane (epipoles at infinity)
    @test all(isfinite, computeresidual(res, PairwisePose(R, SVector(1.0, 0.3, 0.0), KA[1,1], KB[1,1])))
    # The Jacobian is computed without allocating (measured inside a function, so untyped testset variables don't count)
    jacallocs(res, pose) = @allocated computeresjac(static(1), res, pose)
    jacallocs(res, pose)
    @test jacallocs(res, pose) == 0

    # Test optimization from a perturbed pose, with noise-free data
    problem = makeEssentialMatrixproblem(create_twoview_pair(100, R, t, KA, KB))
    problem.variables[1] = update(problem.variables[1], SVector(0.02, -0.03, 0.01, 0.05, -0.05))
    @test all(>(1.e-3), poseerrors(problem.variables[1], R, t))
    result = optimize!(problem, NLLSOptions(; maxiters=100))
    @test result.bestcost < 1.e-10
    @test all(<(1.e-6), poseerrors(problem.variables[1], R, t))

    # Test the optimization wrapper on noisy data
    problem = makeEssentialMatrixproblem(create_twoview_pair(100, R, t, KA, KB; noise=0.5))
    problem.variables[1] = update(problem.variables[1], SVector(0.02, -0.03, 0.01, 0.05, -0.05))
    problem, result, aucs = optimizeEssentialMatrixproblem(problem; maxiters=100)
    @test aucs[2] > aucs[1]
    @test all(<(1.e-2), poseerrors(problem.variables[1], R, t))
end
