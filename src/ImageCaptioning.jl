module ImageCaptioning

using Knet
using Images, FileIO
using Sloth
using JLD2
using MAT
using JSON
using ProgressMeter
using Random
using Statistics

const FloatArray = Union{KnetArray{F}, Array{F}} where F <: AbstractFloat
const FloatMatrix = Union{KnetArray{F,2}, Array{F,2}} where F <: AbstractFloat

const DIR = @__DIR__
EVAL_DIR = mktempdir()

include("vocab.jl"); export Vocabulary
include("data.jl")
include("layers.jl")
include("model.jl")
export ShowAndTell, ShowAttendAndTell
export loss, generate
include("eval.jl")
include("train.jl")


end # module
