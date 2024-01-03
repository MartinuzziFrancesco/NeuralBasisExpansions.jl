using NBeats

@testset "generic_basis Tests" begin
    backcast_size = 10
    forecast_size = 5
    theta = rand(20, backcast_size + forecast_size)  # 20 time series, 15 values each

    basis_function = generic_basis(backcast_size, forecast_size)
    backcast, forecast = basis_function(theta)

    @test size(backcast) == (20, backcast_size)
    @test size(forecast) == (20, forecast_size)
end

@testset "trend_basis Tests" begin
    degree_of_polynomial = 2
    backcast_size = 10
    forecast_size = 5
    theta = rand(20, degree_of_polynomial + 1)  # Ensure theta has the correct size

    basis_function = trend_basis(degree_of_polynomial, backcast_size, forecast_size)
    backcast, forecast = basis_function(theta)

    @test size(backcast) == (20, backcast_size)
    @test size(forecast) == (20, forecast_size)
end

@testset "seasonality_basis Tests" begin
    harmonics = 4
    backcast_size = 10
    forecast_size = 5
    theta = rand(20, harmonics * 4)  # Ensure theta has the correct size

    basis_function = seasonality_basis(harmonics, backcast_size, forecast_size)
    backcast, forecast = basis_function(theta)

    @test size(backcast) == (20, backcast_size)
    @test size(forecast) == (20, forecast_size)
end
