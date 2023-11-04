using ForwardDiff

# Overloads of some specific functions
struct RotMatLie{T}
    x::T
    y::T
    z::T
end

function rodrigues(x::T, y::T, z::T) where T<:ForwardDiff.Dual
    @assert x == 0 && y == 0 && z == 0
    return RotMatLie{T}(x, y, z)
end

du(x) = ForwardDiff.partials(x)
Base.:*(r::AbstractMatrix, u::RotMatLie{T}) where T = SMatrix{3, 3, T, 9}(T(r[1,1], du(u.z)*r[1,2]-du(u.y)*r[1,3]), T(r[2,1], du(u.z)*r[2,2]-du(u.y)*r[2,3]), T(r[3,1], du(u.z)*r[3,2]-du(u.y)*r[3,3]),
                                                                          T(r[1,2], du(u.x)*r[1,3]-du(u.z)*r[1,1]), T(r[2,2], du(u.x)*r[2,3]-du(u.z)*r[2,1]), T(r[3,2], du(u.x)*r[3,3]-du(u.z)*r[3,1]),
                                                                          T(r[1,3], du(u.y)*r[1,1]-du(u.x)*r[1,2]), T(r[2,3], du(u.y)*r[2,1]-du(u.x)*r[2,2]), T(r[3,3], du(u.y)*r[3,1]-du(u.x)*r[3,2]))
Base.:*(u::RotMatLie{T}, r::AbstractMatrix) where T = SMatrix{3, 3, T, 9}(T(r[1,1], du(u.y)*r[3,1]-du(u.z)*r[2,1]), T(r[2,1], du(u.z)*r[1,1]-du(u.x)*r[3,1]), T(r[3,1], du(u.x)*r[2,1]-du(u.y)*r[1,1]),
                                                                          T(r[1,2], du(u.y)*r[3,2]-du(u.z)*r[2,2]), T(r[2,2], du(u.z)*r[1,2]-du(u.x)*r[3,2]), T(r[3,2], du(u.x)*r[2,2]-du(u.y)*r[1,2]),
                                                                          T(r[1,3], du(u.y)*r[3,3]-du(u.z)*r[2,3]), T(r[2,3], du(u.z)*r[1,3]-du(u.x)*r[3,3]), T(r[3,3], du(u.x)*r[2,3]-du(u.y)*r[1,3]))
