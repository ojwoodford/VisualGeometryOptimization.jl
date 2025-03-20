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

function mean0norm1!!(vec::T) where T <: AbstractVector{FD} where FD <: ForwardDiff.Dual{V, P, N} where {V, P, N}
    # Normalize a vector to have zero mean and unit norm
    # Use an algorithm that only loads the data twice
    x = zero(V)
    xx = zero(V)
    p = zeros(SVector{N, P})
    px = zeros(SVector{N, P})
    @simd for v in vec
        x  += ForwardDiff.value(v)
        xx += ForwardDiff.value(v) * ForwardDiff.value(v)
        p += ForwardDiff.partials(v)
        px += ForwardDiff.partials(v) .* ForwardDiff.value(v)
    end
    vecmean = x / length(vec)
    vecnorm = @fastmath V(1) / sqrt(max(xx - vecmean * x, V(1e-5)))
    jacmean = p ./ length(vec)
    px = (px - p .* vecmean) .* vecnorm
    if ismutabletype(T)
        @simd for i in eachindex(vec)
            v = (ForwardDiff.value(vec[i]) - vecmean) .* vecnorm
            p = (ForwardDiff.partials(vec[i]) - jacmean - px .* v) .* vecnorm
            vec[i] = FD(v, ForwardDiff.Partials{N, P}((p...,)))
        end
    else
        # Look at ForwardDiff for how to update
        # v = (v - vecmean) .* vecnorm
        # p = p - jacmean - px .* v
    end
    return vec
end