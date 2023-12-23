module NBeats

using Flux
using LinearAlgebra

abstract type AbstractBlock end
abstract type AbstractBasis end

export Block
include("block.jl")

end
