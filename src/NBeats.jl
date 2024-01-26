module NBeats

using Flux
using LinearAlgebra
using PartialFunctions

abstract type AbstractBlock end

export NBeatsBlock, NBeatsNet
export linear_space, trend_basis, seasonality_basis, generic_basis

include("basis.jl")
include("flux/block.jl")
include("flux/net.jl")

end
