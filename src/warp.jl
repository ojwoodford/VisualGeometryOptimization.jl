using StaticArrays, NLLSsolver, Static

abstract type AbstractProjectiveWarp end
function transform(warp::AbstractProjectiveWarp, coords)
    return coords
end

macro defineprojectivewarp(name, flags)
    ndims = 0
    drp = drn = dsp = dsn = dap = dan = db = ""
    if flags & 1 == 1
        dx = string("updatevec[start+", ndims,  "]")
        ndims += 1
    else
        dx = "0"
    end
    if flags & 2 == 2
        dy = string("updatevec[start+", ndims,  "]")
        ndims += 1
    else
        dy = "0"
    end
    if flags & 4 == 4
        drp = string("+updatevec[start+", ndims,  "]")
        drn = string("+updatevec[start+", ndims,  "]")
        ndims += 1
    end
    if flags & 8 == 8
        dsp = string("+updatevec[start+", ndims,  "]")
        dsn = string("-2*updatevec[start+", ndims,  "]")
        ndims += 1
    end
    if flags & 16 == 16
        dap = string("+updatevec[start+", ndims,  "]")
        dan = string("-2*updatevec[start+", ndims,  "]")
        ndims += 1
    end
    if flags & 32 == 32
        db = string("+updatevec[start+", ndims,  "]")
        ndims += 1
    end
    if flags & 64 == 64
        dc = string("updatevec[start+", ndims,  "]")
        ndims += 1
    else
        dc = "0"
    end
    if flags & 128 == 128
        dd = string("updatevec[start+", ndims,  "]")
        ndims += 1
    else
        dd = "0"
    end

    return esc(quote
        struct $name{T} <: AbstractProjectiveWarp
            mat::SMatrix{3, 3, T, 9}
        end
        $name() = $name(SMatrix{3, 3, Float64, 9}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        NLLSsolver.nvars(::$name) = static($ndims)
        function NLLSsolver.update(var::$name, updatevec, start=1)
            return $name(var.mat * SMatrix{3, 3, eltype(updatevec), 9}(1 $dsp $dap, 0 $drp $db,  $dx, 
                                                                       0 $drn $db,  1 $dsp $dan, $dy, 
                                                                       $dc,         $dd,         1 $dsn))
        end
    end)
end
@defineprojectivewarp(TranslationWarp, 3)
@defineprojectivewarp(ShiftRotateScaleWarp, 15)
@defineprojectivewarp(AffineWarp, 63)
@defineprojectivewarp(HomographyWarp, 255)
