# This example:
#   1. Defines the variable and residual blocks for refinement of an Essential matrix, give point correspondences in a pair of images
#   2. Loads a RANSAC 2020 Essential matrix problem and constructs an NLLSsolver problem from this
#   3. Optimizes the Essential matrix refinement problem and prints the optimization summary, as well as the start and end AUC.
#         AUC = area under curve, specifically the area under the error-recall curve, thresholded at 2 pixel error.

using NLLSsolver, VisualGeometryDatasets, StaticArrays, Static, LinearAlgebra

# Description of Pairwise Pose: the relative pose between two images, with unit vector translation
struct PairwisePose{T}
    poseAtoB::UnitPose3D{T}  # Extrinsic parameters (rotation & unit translation)
    TAp::SVector{2, T} # Projection of translation from A to B
    TBp::SVector{2, T} # Projection of translation from B to A
    sensorA::SimpleCamera{Float64} # Intrinsic sensor parameters (just a single focal length)
    sensorB::SimpleCamera{Float64} # Intrinsic sensor parameters (just a single focal length)
    function PairwisePose(pose::UnitPose3D{T}, sensorA::SimpleCamera{Float64}, sensorB::SimpleCamera{Float64}) where T
        TA = getvec(pose.trans)
        TB = -(pose.rot.m' * TA)
        return new{T}(pose, project(TA), project(TB), sensorA, sensorB)
    end
end
# Construct from the relative rotation R & translation t (X_B = R * X_A + t), and the calibration matrices of each image
PairwisePose(R::AbstractMatrix, t::AbstractVector, fA::Real, fB::Real) = PairwisePose(UnitPose3D(Rotation3DR(SMatrix{3, 3, Float64, 9}(R)), UnitVec3D(SVector{3, Float64}(t))), SimpleCamera(Float64(fA)), SimpleCamera(Float64(fB)))
NLLSsolver.nvars(::PairwisePose) = static(5) # 5 DoF in total (3 rotation, 2 translation)
NLLSsolver.update(var::PairwisePose, updatevec, start=1) = PairwisePose(update(var.poseAtoB, updatevec, start), var.sensorA, var.sensorB)

# Residual that defines the point to epipolar-line error of a correspondence pair
struct EpipolarResidualPair <: AbstractResidual
    xA::SVector{2, Float64} # Point in image A, in metric coordinates
    xB::SVector{2, Float64} # Corresponding point in image B, in metric coordinates
end
NLLSsolver.ndeps(::EpipolarResidualPair) = static(1) # Residual depends on 1 variable
NLLSsolver.nres(::EpipolarResidualPair) = static(2) # Residual vector has length 2 
NLLSsolver.varindices(::EpipolarResidualPair) = 1
NLLSsolver.getvars(::EpipolarResidualPair, vars::AbstractVector) = (vars[@inbounds(1)]::PairwisePose{Float64},)
Base.eltype(::EpipolarResidualPair) = Float64

function NLLSsolver.computeresidual(res::EpipolarResidualPair, pose::PairwisePose)
    # The error is the point to epipolar-line distance in each image, in pixels. There is one error per image
    return SVector(epipolarerror(pose.poseAtoB.rot.m  * push(res.xA, 1), pose.TAp, res.xB, focallength(pose.sensorB)), 
                   epipolarerror(pose.poseAtoB.rot.m' * push(res.xB, 1), pose.TBp, res.xA, focallength(pose.sensorA)))
end

# Define the type of the problem
EssentialMatrixProblem = NLLSProblem{PairwisePose{Float64}, EpipolarResidualPair}

# Function to create a NLLSsolver problem from a RANSAC 2020 Essential matrix dataset
function makeEssentialMatrixproblem(data)::EssentialMatrixProblem
    # Create the problem
    problem = EssentialMatrixProblem()

    # Run a few checks
    @assert(data.K1[3,3] == 1 && data.K2[3,3] == 1, "K[3,3] not 1: currrently unsupported")
    @assert(data.K1[3,1] == 0 && data.K1[3,2] == 0 && data.K2[3,1] == 0 && data.K2[3,2] == 0, "Non-zero bottom row: currently unsupported")
    @assert(data.K1[1,1] == data.K1[2,2] && data.K2[1,1] == data.K2[2,2], "Focal length in x and y directions are different: currently unsupported")
    @assert(data.K1[2,1] == 0 && data.K1[1,2] == 0 && data.K2[2,1] == 0 && data.K2[1,2] == 0, "Non-zero skew: currently unsupported")

    # Add the essential matrix (pairwise pose) block
    addvariable!(problem, PairwisePose(data.R, data.t, data.K1[1,1], data.K2[1,1]))

    # Add the measurement cost blocks
    iKA, iKB = inv(data.K1), inv(data.K2)
    pixel2metric(iK, x) = (y = iK * SVector(x[1], x[2], 1); project(y))
    for (xA, xB) in zip(data.x1, data.x2)
        addcost!(problem, EpipolarResidualPair(pixel2metric(iKA, xA), pixel2metric(iKB, xB)))
    end

    # Return the optimization problem
    return problem
end

function loadEssentialMatrixproblem(name::String, pair::Int)::EssentialMatrixProblem
    # Create the problem
    t = @elapsed begin
        data = loadransac2020twoview(name, :val)[pair]
        problem = makeEssentialMatrixproblem(data)
    end
    show(data)
    println("Data loading and problem construction took ", t, " seconds.")
    return problem
end

# Function to optimize a BAL problem
optimizeEssentialMatrixproblem(name::String, pair::Int; kwargs...) = optimizeEssentialMatrixproblem(loadEssentialMatrixproblem(name, pair); kwargs...)
function optimizeEssentialMatrixproblem(problem::NLLSProblem; projrestype=first(keys(problem.costs.data)), aucthresh=2.0, kwargs...)
    # Compute the starting AUC
    startauc = computeauc(problem, aucthresh, projrestype)
    println("   Start AUC: ", startauc)
    # Optimize the cost
    result = optimize!(problem, NLLSOptions(; reldcost=1.0e-6, kwargs...), nothing, printoutcallback)
    # Compute the final AUC
    endauc = computeauc(problem, aucthresh, projrestype)
    println("   Final AUC: ", endauc)
    # Print out the solver summary
    display(result)
    # Return the optimized problem and optimization results
    return problem, result, (startauc, endauc)
end