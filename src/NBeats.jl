module NBeats

using Flux
using LinearAlgebra

abstract type AbstractBlock end

export NBeatsBlock, NBeatsNet
export GenericBasis, TrendBasis, SeasonalityBasis

include("basis.jl")
include("flux/block.jl")
include("flux/net.jl")

end
