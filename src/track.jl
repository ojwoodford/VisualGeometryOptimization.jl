using VisualGeometryOptimization


struct HomographyTracker <: AbstractTracker
    

    function HomographyTracker(image, coords)
        tracker = new()
        return tracker
    end
end

function track(image, tracker::AbstractTracker, warp)

    return warp
end