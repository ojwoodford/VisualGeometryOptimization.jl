module VisualGeometryOptimization
# Export the API
# Types
export Rotation3DR, Rotation3DL, Point3D, Pose3D, EffPose3D, UnitVec3D, UnitPose3D # 3D geometry variable types
export SimpleCamera, NoDistortionCamera, ExtendedUnifiedCamera, BarrelDistortion, EULensDistortion # Camera sensor & lens variable types
# Functions
export transform, rodrigues, invrodrigues, project, epipolarerror, proj2orthonormal # Multi-view geometry helper functions
export ideal2image, image2ideal, pixel2image, image2pixel, ideal2distorted, distorted2ideal, convertlens # Camera and lens transformations
export optimizeBAproblem, optimizeBALproblem, loadBALproblem, makeBALproblem 
export computeauc # Error metric computation

include("utils.jl")
include("autodiff.jl")
include("camera.jl")	
include("geometry.jl")
include("bundleadjustment.jl")
end
