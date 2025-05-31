using NLLSsolver, StaticArrays

# Patches are structures that contain coordinates and image intensities
# World patches contain world coordinates, image patches contain image coordinates
struct StaticWorldPatchNCC{TC, TV, N, DC, DV}
    coords::StaticMatrix{N, DC, TC, N*DC}
    values::StaticMatrix{N, DV, TV, N*DV}
end

isnormalized(::StaticWorldPatchNCC) = true

# A PatchResidual links a source patch to a target image
struct PatchResidual{TP, TI} <: NLLSsolver.AbstractResidual
    patch::Ref{TP}
    image::Ref{TI}
end

# Compute a photometric error
function NLLSsolver.computeresidual(patchres::PatchResidual, warp)
    # Compute the patch image coordinates
    imcoords = transform(warp, patchres.patch[].coords)
    # Sample the image
    residual = sample(patchres.image[], imcoords)
    # Normalize if necessary
    if isnormalized(patchres.patch[])
        residual = mean0norm1!!(residual)
    end
    # Subtract the target patch
    if ismutabletype(typeof(residual))
        residual .-= patchres.patch[].values
    else
        residual = residual .- patchres.patch[].values
    end
    # Return the residual
    return residual
end
