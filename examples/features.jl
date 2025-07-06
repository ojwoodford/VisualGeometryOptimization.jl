using TestImages, Images, GLMakie, VisualGeometryOptimization, StaticArrays, TensorOperations

const edgelet = SMatrix{2, 16, Float64, 32}([0.0 0.0 0.0 0.5 -0.5 -1.0 0.0 1.0 1.0 0.0 -1.0 -0.5 0.5 0.0 0.0 0.0;
                                             6.0 4.0 2.5 1.5 1.5 0.5 0.5 0.5 -0.5 -0.5 -0.5 -1.5 -1.5 -2.5 -4.0 -6.0])

function render_points_lines(axes, points, colors=:blue; marker=:circle, markersize=6, linestyle=:solid, linewidth=1, colormap=:viridis)
    sz = (size(points)..., 1, 1, 1)
    @assert(sz[3] == 2 || sz[3] == 3, "Points must be 2D or 3D")
    if ~isnothing(linestyle) && sz[1] > 1
        render_lines(axes, points, colors, sz, linestyle, linewidth, colormap)
    end
    if ~isnothing(marker)
        render_points(axes, points, colors, sz, marker, markersize, colormap)
    end
end

function render_lines(axes, points, colors, sz, linestyle, linewidth, colormap)
    # Render the lines
    nans = fill(NaN, 1, sz[2])
    X = vec(vcat(points[:,:,1], nans))
    Y = vec(vcat(points[:,:,2], nans))
    C = colors isa AbstractArray ? vec(vcat(colors, view(colors, 1, :))) : colors
    if sz[3] == 3
        lines!(axes, X, Y, vec(vcat(points[:,:,3], nans)); linestyle=(linestyle, :dense), linewidth=linewidth, colormap=colormap, color=C)
    else
        lines!(axes, X, Y; linestyle=(linestyle, :dense), linewidth=linewidth, colormap=colormap, color=C)
    end
end

function render_points(axes, points, colors, sz, marker, markersize, colormap)
    # Render the points
    if sz[3] == 3
        scatter!(axes, vec(view(points, :, :, 1)), vec(view(points, :, :, 2)), vec(view(points, :, :, 3)); marker=marker, markersize=markersize, depthsorting=true, colormap=colormap, color=colors)
    else
        scatter!(axes, vec(view(points, :, :, 1)), vec(view(points, :, :, 2)); marker=marker, markersize=markersize, colormap=colormap, color=colors)
    end
end

function visualize_edgelets(im)
    # Compute the edgelet points
    im = Gray.(im)
    X, rot, gradmag = extract_edgelets(im)

    # Transform the edgelets
    rot = rot ./ maximum(abs, rot, dims=1)
    rot = reshape(rot[SVector(2, 1, 1, 2),:], (2, 2, :))
    rot[1,1,:] .= -rot[1,1,:]
    X = repeat(reshape(X, 2, 1, :), outer=(1, 16, 1))
    @tensor points[j,k,i] := rot[i,l,k] * edgelet[l,j] + X[i,j,k]

    # Display the image and points
    fig = Figure()
    ax = GLMakie.Axis(fig[1, 1], aspect = DataAspect(), yreversed = true)
    image!(ax, im'; interpolate=false)
    render_points_lines(ax, points, :green; linestyle=nothing)
    points = points[[1, 3, 4, 8, 9, 13, 14, 16, 14, 12, 11, 6, 6, 3],:,:]
    render_points_lines(ax, points, :green; marker=nothing)
    display(fig)
    return nothing
end

visualize_edgelets(testimage("cameraman"))
