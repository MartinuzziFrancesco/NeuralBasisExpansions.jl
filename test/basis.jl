using NBeats

backcast_size = 10
forecast_size = 5
degree_of_polynomial = 3
harmonics = 2

@testset "GenericBasis Tests" begin
    theta = rand(20, backcast_size + forecast_size)  # 20 time series, 15 values each

    basis_function = GenericBasis(backcast_size, forecast_size)
    backcast, forecast = basis_function(theta)

    @test size(backcast) == (20, backcast_size)
    @test size(forecast) == (20, forecast_size)
end

@testset "TrendBasis Tests" begin
    theta = rand(20, degree_of_polynomial + 1)

    basis_function = TrendBasis(degree_of_polynomial, backcast_size, forecast_size)
    backcast, forecast = basis_function(theta)

    @test size(backcast) == (20, backcast_size)
    @test size(forecast) == (20, forecast_size)
end

@testset "SeasonalityBasis Tests" begin
    theta = rand(20, 4)

    basis_function = SeasonalityBasis(harmonics, backcast_size, forecast_size)
    backcast, forecast = basis_function(theta)

    @test size(backcast) == (20, backcast_size)
    @test size(forecast) == (20, forecast_size)
end
