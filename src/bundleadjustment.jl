# This example:
#   1. Defines the variable and residual blocks for "Bundle Adjustment in the Large" (BAL) problems
#   2. Loads a BAL dataset and constructs an NLLSsolver problem from this
#   3. Optimizes the bundle adjustment problem and prints the optimization summary, as well as the start and end AUC.
#         AUC = area under curve, specifically the area under the error-recall curve, thresholded at 2 pixel error.

using NLLSsolver, VisualGeometryDatasets, StaticArrays, Static, LinearAlgebra

# Description of BAL image, and function to transform a landmark from world coordinates to pixel coordinates
struct BALImage{T}
    pose::EffPose3D{T}   # Extrinsic parameters (rotation & translation)
    sensor::SimpleCamera{T} # Intrinsic sensor parameters (just a single focal length)
    lens::BarrelDistortion{T} # Intrinsic lens parameters (k1 & k2 barrel distortion)
end
BALImage(rx::T, ry::T, rz::T, tx::T, ty::T, tz::T, f::T, k1::T, k2::T) where T<:Real = BALImage{T}(EffPose3D(Pose3D(rx, ry, rz, tx, ty, tz)), SimpleCamera(f), BarrelDistortion(k1, k2))
BALImage(v) = BALImage(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])
NLLSsolver.nvars(::BALImage) = static(9) # 9 DoF in total for the image (6 extrinsic, 3 intrinsic)
NLLSsolver.update(var::BALImage, updatevec, start=1) = BALImage(update(var.pose, updatevec, start), update(var.sensor, updatevec, start+6), update(var.lens, updatevec, start+7))

# Residual that defines the reprojection error of a BAL measurement
BALResidual{T} = SimpleError2{2, T, BALImage{T}, Point3D{T}}
BALResidual(m, v) = BALResidual{eltype(m)}(SVector(m[1], m[2]), SVector(Int(v[1]), Int(v[2])))
NLLSsolver.generatemeasurement(im::BALImage, X::Point3D) = ideal2image(im.sensor, ideal2distorted(im.lens, -project(im.pose * X)))
const balrobustifier = HuberKernel(2.)
NLLSsolver.robustkernel(::BALResidual) = balrobustifier

function NLLSsolver.computeresjac(::StaticInt{3}, residual::BALResidual, im, point)
    # Exploit the parameterization to make the jacobian computation more efficient
    res, jac = computeresjac(static(1), residual, im, point)
    return res, hcat(jac, @inbounds(-view(jac, :, NLLSsolver.SR(4, 6))))
end

# Define the type of the problem
BALProblem = NLLSProblem{Union{BALImage{Float64}, Point3D{Float64}}, BALResidual{Float64}}

# Function to create a NLLSsolver problem from a BAL dataset
function makeBALproblem(data)::BALProblem
    # Create the problem
    problem = BALProblem()

    # Add the camera variable blocks
    for cam in data.cameras
        addvariable!(problem, BALImage(cam))
    end
    numcameras = length(data.cameras)
    # Add the landmark variable blocks
    for lm in data.landmarks
        addvariable!(problem, Point3D(lm[1], lm[2], lm[3]))
    end

    # Add the measurement cost blocks
    for meas in data.measurements
        addcost!(problem, BALResidual(SVector(meas.x, meas.y), SVector(meas.camera, meas.landmark + numcameras)))
    end

    # Return the optimization problem
    return problem
end

function loadBALproblem(name)::BALProblem
    # Create the problem
    t = @elapsed begin
        data = loadbaldataset(name)
        problem = makeBALproblem(data)
    end
    show(data)
    println("Data loading and problem construction took ", t, " seconds.")
    return problem
end

# Function to optimize a BAL problem
optimizeBALproblem(name::String; kwargs...) = optimizeBALproblem(loadBALproblem(name); kwargs...)
function optimizeBALproblem(problem::NLLSProblem; kwargs...)
    # Compute the starting AUC
    startauc = computeauc(problem, 2.0, problem.costs.data[BALResidual{Float64}])
    println("   Start AUC: ", startauc)
    # Optimize the cost
    result = optimize!(problem, NLLSOptions(; reldcost=1.0e-6, kwargs...), nothing, printoutcallback)
    # Compute the final AUC
    endauc = computeauc(problem, 2.0, problem.costs.data[BALResidual{Float64}])
    println("   Final AUC: ", endauc)
    # Print out the solver summary
    display(result)
    # Return the optimized problem and optimization results
    return problem, result, (startauc, endauc)
end
