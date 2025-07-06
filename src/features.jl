using StaticArrays, Images

function imnonmaxsuppression(mag, nonmaxradius::Float64=1.5)
    # Create the mask
    el = ceil(nonmaxradius)
    el = LinRange(-el, el, 2*Int(el)+1) .^ 2
    el = repeat(el, outer=(1, length(el))) + repeat(el', outer=(length(el), 1))
    el = el .<= (nonmaxradius * nonmaxradius)
    # Dilate and match
    return dilate(mag, el) .== mag
end

function compute_order(X, W, n, radius=0.0)
sz = size(X)
n = min(n, sz[2])
radius = radius * radius
D = fill(Inf, sz[2])
W = W .* W
m, i = findmax(W)
order = sizehint!(Int[], n)
push!(order, i)
D2 = 0.5 * vec(sum(X .* X, dims=1))
for j in 1:n
    D = min.(D, (D2 - (X[:,i]' * X)') .+ D2[i])
    m, i = findmax(D .* W)
    if m == 0 || D[i] < radius
        break
    end
    push!(order, i)
end
return order
end

function extract_edgelets(im; mingradmag::Float64=0.01, mask::BitMatrix=trues(size(im)), nonmaxradius::Float64=10.0)
    # Compute image gradients
    Iy, Ix, gradmag, orient = imedge(im, KernelFactors.bickley, "replicate")

    # Thin the edges
    gradmag, subpix = thin_edges_nonmaxsup_subpix(gradmag, orient)

    # Mask off pixels that don't meet the selection criteria
    # Mask edge pixels
    mask[1:6,:] .= false
    mask[end-5:end,:] .= false
    mask[:,1:6] .= false
    mask[:,end-5:end] .= false
    # Mask pixels with low gradient magnitude
    mask .&= gradmag .> mingradmag
    # Mask pixels that are not local maxima
    gradmag = gradmag .* mask
    mask .&= imnonmaxsuppression(gradmag, nonmaxradius/2)
    # Vectorize the outputs
    X = reshape(stack(v -> (v.x, v.y), subpix[mask]), 2, :)
    gradmag = gradmag[mask]
    dir = vcat(Ix[mask]', Iy[mask]')
    # Sort according to gradient magnitude and distance
    order = compute_order(X, log1p.(max.(gradmag, 0.0)), length(gradmag), nonmaxradius)
    X = X[:,order]
    dir = dir[:,order]
    gradmag = gradmag[order]
    return (X, dir, gradmag)
end
