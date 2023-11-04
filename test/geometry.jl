using VisualGeometryOptimization, NLLSsolver, LinearAlgebra, StaticArrays, Test

checkduals(x, y) = NLLSsolver.extractvaldual(vec(x)) == NLLSsolver.extractvaldual(vec(y))
const getvec = VisualGeometryOptimization.getvec
const inverse = VisualGeometryOptimization.inverse

@testset "geometry.jl" begin
    # Test utility functions
    O = proj2orthonormal(randn(5, 5))
    @test O' * O ≈ I

    # Test SO(3) conversions
    @test rodrigues(0., 0., 0.) == I
    @test rodrigues(Float64(π), 0., 0.) ≈ Diagonal(SVector(1., -1., -1.))
    sv(x, y, z) = SVector(Float64(x), Float64(y), Float64(z))
    testrots = [sv(0, 0, 0), sv(π, 0, 0), sv(0, π, 0), sv(0, 0, π)]
    for v in testrots
        @test invrodrigues(rodrigues(v[1], v[2], v[3])) ≈ v
    end
    for i in 1:100
        R = proj2orthonormal(randn(SMatrix{3, 3, Float64, 9}) * 10)
        if det(R) < 0
            R *= diagm(SVector(-1.0, 1.0, 1.0))
        end
        @test rodrigues(invrodrigues(R)) ≈ R
    end

    # Test points
    updatevec = zeros(3)
    point = update(Point3D(normalize(SVector(0.9, 1.1, -1.0))), updatevec)
    pointu = update(UnitVec3D(getvec(point)), updatevec)
    @test project(point) == SVector(-0.9, -1.1)
    @test project(pointu) == SVector(-0.9, -1.1)
    @test point - pointu == Point3D()

    # Test rotations
    updatevec = randn(3)
    rotr = update(Rotation3DR(), updatevec)
    rotl = update(Rotation3DL(), -updatevec)
    @test getvec((rotl * rotr) * point) ≈ getvec(point)
    @test getvec((inverse(rotl) * inverse(rotr)) * point) ≈ getvec(point)

    # Test rotation autodiff updates
    updatevec = NLLSsolver.dualzeros(Float64, Val(4))
    T = eltype(updatevec)
    updatemat = SMatrix{3, 3, T, 9}(T(1), updatevec[3], -updatevec[2], -updatevec[3], T(1), updatevec[1], updatevec[2], -updatevec[1], T(1))
    rotr = Rotation3DR(randn(), randn(), randn())
    rotl = Rotation3DL(randn(), randn(), randn())
    @test checkduals(update(rotr, updatevec).m, rotr.m * updatemat)
    @test checkduals(update(rotl, updatevec).m, updatemat * rotl.m)

    # Test poses
    updatevec = zeros(6)
    pose = update(Pose3D(rotr, point), updatevec)
    poseu = update(UnitPose3D(rotr, pointu), updatevec)
    @test getvec(pose * (inverse(poseu) * point)) ≈ getvec(point)
    @test getvec(poseu * (inverse(pose) * point)) ≈ getvec(point)
    posee = update(EffPose3D(pose), updatevec)
    @test getvec(inverse(pose) * (posee * point)) ≈ getvec(point)
    @test getvec(pose * (inverse(posee) * point)) ≈ getvec(point)
end
