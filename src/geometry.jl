using StaticArrays, LinearAlgebra, NLLSsolver
import IfElse: ifelse

rodrigues(v) = rodrigues(v[1], v[2], v[3])
function rodrigues(x::T, y::T, z::T) where T<:Number
    theta2 = x * x + y * y + z * z
    cosf = T(0.5)
    sinc = T(1)
    if theta2 > 2.23e-16
        theta = sqrt(theta2)
        sinc, cosf = sincos(theta)
        cosf -= 1
        sinc /= theta
        cosf /= -theta2
    end
    a = x * y * cosf
    b = sinc * z
    c = x * z * cosf
    d = sinc * y
    e = y * z * cosf
    f = sinc * x
    return SMatrix{3, 3, T, 9}((x * x - theta2) * cosf + 1, a + b, c - d,
                                a - b, (y * y - theta2) * cosf + 1, e + f,
                                c + d, e - f, (z * z - theta2) * cosf + 1)
end

function invrodrigues(R)
    @assert Size(R) == (3, 3)
    diagR = diag(R)
    traceR = sum(diagR)
    m, n = findmax(diagR)
    traceR_ = m * 2 - traceR
    tf = traceR >= traceR_
    n = ifelse(tf, 0, n)
    traceR = ifelse(tf, traceR, traceR_) + 1
    theta = sqrt(traceR)
    if n < 2
        if n == 0
            # n == 0
            X = @inbounds SVector(R[3,2] - R[2,3], R[1,3] - R[3,1], R[2,1] - R[1,2])
            theta /= 2
        else
            # n == 1
            X = @inbounds SVector(traceR, R[2,1] + R[1,2], R[1,3] + R[3,1])
            theta = (R[3,2] - R[2,3]) / (theta * 2)
        end
    else
        if n == 2
            # n == 2
            X = @inbounds SVector(R[2,1] + R[1,2], traceR, R[3,2] + R[2,3])
            theta = (R[1,3] - R[3,1]) / (theta * 2)
        else
            # n == 3
            X = @inbounds SVector(R[1,3] + R[3,1], R[3,2] + R[2,3], traceR)
            theta = (R[2,1] - R[1,2]) / (theta * 2)
        end
    end
    sqnorm = X' * X
    if sqnorm == 0
        return X
    end
    return X * (2 * acos(theta) / sqrt(sqnorm))
end

function proj2orthonormal(M)
    s = svd(M);
    return s.U * s.V';
end

abstract type AbstractPoint3D end
struct Point3D{T<:Real} <: AbstractPoint3D
    v::SVector{3, T}
end
Point3D(x::T, y::T, z::T) where T = Point3D{T}(SVector{3, T}(x, y, z))
Point3D() = Point3D{Float64}()
Point3D{T}() where T = Point3D{T}(SVector{3, T}(0, 0, 0))
NLLSsolver.nvars(::Point3D) = static(3)
NLLSsolver.update(var::Point3D, updatevec, start=1) = Point3D(var.v + view(updatevec, NLLSsolver.SR(0, 2) .+ start))
@inline getvec(x::Point3D) = x.v
@inline getind(x::Point3D, ind) = x.v[ind]

abstract type AbstractRotation3D end
struct Rotation3DR{T<:Real} <: AbstractRotation3D
    m::SMatrix{3, 3, T, 9}
end
Rotation3DR(x, y, z) = Rotation3DR(rodrigues(x, y, z))
Rotation3DR() = Rotation3DR(SMatrix{3, 3, Float64}(1., 0., 0., 0., 1., 0., 0., 0., 1.))
NLLSsolver.nvars(::Rotation3DR) = static(3)
NLLSsolver.update(var::Rotation3DR, updatevec, start=1) = Rotation3DR(var.m * rodrigues(updatevec[start], updatevec[start+1], updatevec[start+2]))

struct Rotation3DL{T<:Real} <: AbstractRotation3D
    m::SMatrix{3, 3, T, 9}
end
Rotation3DL(x, y, z) = Rotation3DL(rodrigues(x, y, z))
Rotation3DL() = Rotation3DL(SMatrix{3, 3, Float64}(1., 0., 0., 0., 1., 0., 0., 0., 1.))
NLLSsolver.nvars(::Rotation3DL) = static(3)
NLLSsolver.update(var::Rotation3DL, updatevec, start=1) = Rotation3DL(rodrigues(updatevec[start], updatevec[start+1], updatevec[start+2]) * var.m)

struct UnitVec3D{T<:Real} <: AbstractPoint3D
    v::Rotation3DR{T}
end
UnitVec3D() = UnitVec3D(Rotation3DR())
function UnitVec3D(x::SVector{3, T}) where T
    # Initialize y & z axes
    ind = findmin(abs, x)[2]
    y = cross(SVector(T(1==ind), T(2==ind), T(3==ind)), x)
    z = cross(x, y)
    # Normalize all the vectors
    return UnitVec3D{T}(Rotation3DR(hcat(normalize(x), normalize(y), normalize(z))))
end
UnitVec3D(x::T, y::T, z::T) where T = UnitVec3D(SVector{3, T}(x, y, z))
NLLSsolver.nvars(::UnitVec3D) = static(2)
NLLSsolver.update(var::UnitVec3D, updatevec, start=1) = UnitVec3D(update(var.v, SVector(0, updatevec[start], updatevec[start+1])))
getvec(x::UnitVec3D) = @inbounds(view(x.v.m, :, 1))
getind(x::UnitVec3D, ind) = x.v.m[ind,1]


abstract type AbstractPose3D end
struct Pose3D{T<:Real} <: AbstractPose3D
    rot::Rotation3DR{T}
    trans::Point3D{T}
end
Pose3D(rx, ry, rz, tx, ty, tz) = Pose3D(Rotation3DR(rx, ry, rz), Point3D(tx, ty, tz))
Pose3D() = Pose3D(Rotation3DR(), Point3D())
NLLSsolver.nvars(::Pose3D) = static(6)
NLLSsolver.update(var::Pose3D, updatevec, start=1) = Pose3D(update(var.rot, updatevec, start), update(var.trans, updatevec, start+3))

struct UnitPose3D{T<:Real} <: AbstractPose3D
    rot::Rotation3DR{T}
    trans::UnitVec3D{T}
end
UnitPose3D() = Pose3D(Rotation3DR(), UnitVec3D())
UnitPose3D(rx::T, ry::T, rz::T, tx::T, ty::T, tz::T) where T = Pose3D(Rotation3DR(rx, ry, rz), UnitVec3D(tx, ty, tz))
NLLSsolver.nvars(::UnitPose3D) = static(5)
NLLSsolver.update(var::UnitPose3D, updatevec, start=1) = UnitPose3D(update(var.rot, updatevec, start), update(var.trans, updatevec, start+3))

struct EffPose3D{T<:Real} <: AbstractPose3D
    rot::Rotation3DL{T}
    camcenter::Point3D{T}
end
EffPose3D(rx, ry, rz, cx, cy, cz) = EffPose3D(Rotation3DL(rx, ry, rz), Point3D(cx, cy, cz))
EffPose3D() = EffPose3D(Rotation3DL(), Point3D())
NLLSsolver.nvars(::EffPose3D) = static(6)
NLLSsolver.update(var::EffPose3D, updatevec, start=1) = EffPose3D(update(var.rot, updatevec, start), update(var.camcenter, updatevec, start+3))
inverse(var::EffPose3D) = Pose3D(Rotation3DR(var.rot.m'), var.camcenter)
transform(pose::EffPose3D, point::AbstractPoint3D) = Point3D(pose.rot.m * (getvec(point) - pose.camcenter.v))

# Pose conversions
Pose3D(pose::EffPose3D) = Pose3D(Rotation3DR(pose.rot.m), Point3D(pose.rot.m * -pose.camcenter.v))
Pose3D(pose::Pose3D) = pose
EffPose3D(pose::Pose3D) = EffPose3D(Rotation3DL(pose.rot.m), Point3D(pose.rot.m' * -pose.trans.v))
EffPose3D(pose::EffPose3D) = pose

# Transformations on abstract types
project(x::AbstractPoint3D) = SVector(getind(x, 1), getind(x, 2)) ./ getind(x, 3)
inverse(rot::T) where T<:AbstractRotation3D = T(rot.m')
inverse(var::AbstractPose3D) = Pose3D(Rotation3DR(var.rot.m'), Point3D(var.rot.m' * -getvec(var.trans)))
transform(rota::T, rotb::AbstractRotation3D) where T<:AbstractRotation3D = T(rota.m * rotb.m)
transform(rot::AbstractRotation3D, point::AbstractPoint3D) = Point3D(rot.m * getvec(point))
transform(pose::AbstractPose3D, point::AbstractPoint3D) = pose.rot * point + pose.trans

# Overload arithmetic operators
Base.:*(rota::AbstractRotation3D, rotb::AbstractRotation3D) = transform(rota, rotb)
Base.:*(rot::AbstractRotation3D, point::AbstractPoint3D) = transform(rot, point)
Base.:*(posea::AbstractPose3D, poseb::AbstractPose3D) = transform(posea, poseb)
Base.:*(pose::AbstractPose3D, point::AbstractPoint3D) = transform(pose, point)
Base.:+(pointa::AbstractPoint3D, pointb::AbstractPoint3D) = Point3D(getvec(pointa) + getvec(pointb))
Base.:-(pointa::AbstractPoint3D, pointb::AbstractPoint3D) = Point3D(getvec(pointa) - getvec(pointb))

function epipolarerror(RX, T, x, W=nothing)
    Tp = project(T)
    xe = x - Tp
    RXp = project(RX) - Tp
    RXp = SVector(RXp[2], -RXp[1])
    if !isnothing(W)
        RXp = W' \ RXp
        xe = W * xe
    end
    return dot(xe, normalize(RXp))
end
