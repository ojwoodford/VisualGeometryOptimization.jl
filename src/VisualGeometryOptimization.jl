module VisualGeometryOptimization
# Export the API
# Types
export Rotation3DR, Rotation3DL, Point3D, Pose3D, EffPose3D, UnitVec3D, UnitPose3D # 3D geometry variable types
export SimpleCamera, NoDistortionCamera, ExtendedUnifiedCamera, BarrelDistortion, EULensDistortion # Camera sensor & lens variable types
export HomographyWarp, AffineWarp, ShiftRotateScaleWarp, TranslationWarp # Warp variable types
export ImageScale, Image, ImagePyramid # Image variable types
# Functions
export transform, rodrigues, invrodrigues, project, epipolarerror, proj2orthonormal # Multi-view geometry helper functions
export mean0norm1!! # Normalization functions
export ideal2image, image2ideal, ideal2distorted, distorted2ideal, convertlens # Camera and lens transformations
export sample, image2pixel, pixel2image, halfsize, baseimage # Image sampling and transformations
export optimizeBAproblem, optimizeBALproblem, loadBALproblem 

include("utils.jl")
include("autodiff.jl")
include("camera.jl")	
include("geometry.jl")
include("warp.jl")
include("image.jl")
include("normalize.jl")
include("bundleadjustment.jl")
end
