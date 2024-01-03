module NBeats

using Flux
using LinearAlgebra

abstract type AbstractBlock end

export NBeatsBlock, NBeatsNet
export generic_basis, trend_basis, seasonality_basis

include("block.jl")
include("basis.jl")

end
