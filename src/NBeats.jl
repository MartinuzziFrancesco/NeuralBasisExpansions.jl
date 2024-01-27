module NBeats

using Flux
using LinearAlgebra
using PartialFunctions

#abstract type AbstractBlock end

export NBeatsBlock, NBeatsNet
export BasisLayer
export linear_space, trend_basis, seasonality_basis, generic_basis
export split, train!, predict

include("basis.jl")
include("block.jl")
include("net.jl")

end
