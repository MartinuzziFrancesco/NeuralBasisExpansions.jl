using NeuralBasisExpansions

const layer_size = 128
const theta_size = 4
const backcast_length = 10
const forecast_length = 5

all_basis = [generic_basis, trend_basis, seasonality_basis]

@testset "BasisLayer output size tests" begin
    for basis_function in all_basis
        bl = BasisLayer(
            layer_size,
            theta_size,
            basis_function;
            backcast_length=backcast_length,
            forecast_length=forecast_length,
        )
        rr = bl(rand(Float32, layer_size))
        @test length(rr[1]) == backcast_length
        @test length(rr[2]) == forecast_length
    end
end
