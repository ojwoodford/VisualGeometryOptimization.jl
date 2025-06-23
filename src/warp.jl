using StaticArrays, NLLSsolver, Static

abstract type AbstractProjectiveWarp end
transform(warp::AbstractProjectiveWarp, coords::SMatrix{2}) = (view(warp.mat, 1:2, 1:2) * coords .+ view(warp.mat, 1:2, 3)) ./ (view(warp.mat, 3, 1:2)' * coords .+ view(warp.mat, 3, 3))
transform(warp::AbstractProjectiveWarp, coords::SMatrix{3}) = (view(warp.mat, 1:2, :) * coords) ./ (view(warp.mat, 3, :)' * coords)

macro defineprojectivewarp(name, flags)
    ndims = 0
    if flags & 1 == 1
        d13 = :(updatevec[start+$ndims])
        ndims += 1
    else
        d13 = 0
    end
    if flags & 2 == 2
        d23 = :(updatevec[start+$ndims])
        ndims += 1
    else
        d23 = 0
    end
    if flags & 4 == 4
        d12 = :(-updatevec[start+$ndims])
        d21 = :(updatevec[start+$ndims])
        ndims += 1
    else
        d12 = 0
        d21 = 0
    end
    if flags & 8 == 8
        d11 = :(1+updatevec[start+$ndims])
        d22 = :(1+updatevec[start+$ndims])
        d33 = :(1-2*updatevec[start+$ndims])
        ndims += 1
    else
        d11 = 1
        d22 = 1
        d33 = 1
    end
    if flags & 16 == 16
        d11 = :($d11+updatevec[start+$ndims])
        d22 = :($d22-updatevec[start+$ndims])
        ndims += 1
    end
    if flags & 32 == 32
        d12 = :($d12+updatevec[start+$ndims])
        d21 = :($d21+updatevec[start+$ndims])
        ndims += 1
    end
    if flags & 64 == 64
        d31 = :(updatevec[start+$ndims])
        ndims += 1
    else
        d31 = 0
    end
    if flags & 128 == 128
        d32 = :(updatevec[start+$ndims])
        ndims += 1
    else
        d32 = 0
    end

    return esc(quote
        struct $name{T} <: AbstractProjectiveWarp
            mat::SMatrix{3, 3, T, 9}
        end
        $name() = $name(SMatrix{3, 3, Float64, 9}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        NLLSsolver.nvars(::$name) = static($ndims)
        function NLLSsolver.update(var::$name, updatevec, start=1)
            return $name(var.mat * SMatrix{3, 3, eltype(updatevec), 9}($d11, $d12, $d13, 
                                                                       $d21, $d22, $d23, 
                                                                       $d31, $d32, $d33))
        end
    end)
end
@defineprojectivewarp(TranslationWarp, 3)
@defineprojectivewarp(ShiftRotateScaleWarp, 15)
@defineprojectivewarp(AffineWarp, 63)
@defineprojectivewarp(HomographyWarp, 255)
