module NBeats

using Flux
using LinearAlgebra

abstract type AbstractBlock end

export Block
include("block.jl")

end
