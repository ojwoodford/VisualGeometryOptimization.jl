
function vecmeannorm(vec::AbstractVector{T}) where T
    # Compute the mean and normalizer of a vector, in a single pass
    x = zero(T)
    xx = zero(T)
    @simd for v in vec
        x  += v
        xx += v * v
    end
    vecmean = x / length(vec)
    vecnorm = @fastmath T(1) / sqrt(max(xx - vecmean * x, T(1e-5)))
    return (vecmean, vecnorm)
end

function mean0norm1!!(vec::T) where T
    # Normalize a vector to have zero mean and unit norm
    # Use an algorithm that only loads the data twice
    vecmean, vecnorm = vecmeannorm(vec)
    if ismutabletype(T)
        vec .= (vec .- vecmean) .* vecnorm
    else
        vec = (vec .- vecmean) .* vecnorm
    end
    return vec
end

function mean0norm1!!(vec::T, orig) where T
    # Normalize a vector to have zero mean and unit norm, then subtract another vector
    # Use an algorithm that only loads the data twice
    vecmean, vecnorm = vecmeannorm(vec)
    if ismutabletype(T)
        vec .= (vec .- vecmean) .* vecnorm .- orig
    else
        vec = (vec .- vecmean) .* vecnorm .- orig
    end
    return vec
end
