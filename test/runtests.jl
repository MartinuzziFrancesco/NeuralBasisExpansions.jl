using NBeats
using Test
using Aqua
using JET

@testset "NBeats.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NBeats)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(NBeats; target_defined_modules = true)
    end
    # Write your tests here.
end
