module ImageCaptioning

using Knet
using Sloth
using Images
using JLD2
using MAT
using JSON
using FileIO
using Random

const FloatArray = Union{KnetArray{F}, Array{F}} where F <: AbstractFloat
const FloatMatrix = Union{KnetArray{F,2}, Array{F,2}} where F <: AbstractFloat

include("vocab.jl"); export Vocabulary
include("data.jl")
include("model.jl"); export ShowAndTell


end # module
