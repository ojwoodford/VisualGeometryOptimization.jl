using Images, ImageTransformations, Interpolations, StaticArrays

# ImageScale is a structure that contains the offset and scale of an image
struct ImageScale{T<:Real}
    offset::SVector{2, T}
    scale::T
    invscale::T
    ImageScale(offset::SVector{2, T}, scale::T) where T = new{T}(offset, scale, 1 / scale)
end
ImageScale(xoff::T, yoff::T, scale::T) where T = ImageScale(SVector(xoff, yoff), scale)
ImageScale(offset::SVector{2, T}) where T = ImageScale(offset, offset[1])

# ImageScale is a structure that contains the offset and scale of an image
halfsize(imscale::ImageScale) = ImageScale(imscale.offset .* 0.5, imscale.scale * 0.5)

# Pixel to scale-invariant image coordinate conversion
image2pixel(imscale, x) = x .* imscale.scale .+ imscale.offset
pixel2image(imscale, x) = (x .- imscale.offset) .* imscale.invscale
pixel2image(imscale, x, W) = ((x .- imscale.offset) .* imscale.invscale, W .* imscale.scale)

struct Image{TI<:AbstractInterpolation, TS}
    iminterpolator::TI
    imscale::ImageScale{TS}
    Image(iminterpolator::TI, imscale::ImageScale{TS}) where {TI <: AbstractInterpolation, TS} = new{TI, TS}(iminterpolator, imscale)
end
Image(im::TI, imscale::ImageScale{TS}) where {TI <: AbstractMatrix, TS} = Image(ImageTransformations.box_extrapolation(im), imscale)
Image(im::AbstractMatrix, xoff, yoff, scale) = Image(im, ImageScale(xoff, yoff, scale))
Base.size(im::Image) = Base.size(im.iminterpolator)

# Sample: convert to pixel coordinates, then call the interpolator
sample(iminterpolator::AbstractInterpolation, pixcoords::SVector{2, T}) where T <: Real = iminterpolator(pixcoords[1], pixcoords[2])
sample(image::Image, imcoords::SVector{2, T}) where T <: Real = sample(image.iminterpolator, image2pixel(image.imscale, imcoords))
sample(image::Image, imcoords::Vector{SVector{2, T}}) where T <: Real = [sample(image, imcoord) for imcoord in imcoords]
sample(image::Image, imcoords::SVector{N, SVector{2, T}}) where {N, T <: Real} = SVector(ntuple(i -> sample(image, imcoords[i]), Val(N)))

# Downsample the image by a factor of 2
baseimage(image::Image) = image.iminterpolator.itp.coefs
halfsize(image::Image) = Image(imresize(baseimage(image), ratio=0.5), halfsize(image.imscale))

# Structure for storing an image pyramid
struct ImagePyramid{TI, TS}
    level::Vector{Image{TI, TS}}
    function ImagePyramid(image::Image{TI, TS}) where {TI, TS}
        # Compute the number of levels
        nlevels = max(Int(floor(log2(min(size(image)...)) - 4)), 1)
        # Construct the levels
        level = Vector{Image{TI, TS}}(undef, nlevels)
        level[1] = image
        for i in 2:nlevels
            # Downsample the image
            image = halfsize(image)
            # Store in the vector
            level[i] = image
        end
        # Return the image pyramid
        return new{TI, TS}(level)
    end
end
