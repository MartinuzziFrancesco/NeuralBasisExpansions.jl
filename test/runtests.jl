using Test
using SafeTestsets

@safetestset "Quality Assurance" begin
    include("qa.jl")
end

@safetestset "Basis" begin
    include("basis.jl")
end
