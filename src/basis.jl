struct GenericBasis
    backcast_size::Int
    forecast_size::Int
end

function (basis::GenericBasis)(theta::AbstractArray)
    backcast = theta[:, 1:(basis.backcast_size)]
    forecast = theta[:, (end - basis.forecast_size + 1):end]
    return backcast, forecast
end

struct TrendBasis
    polynomial_size::Int
    backcast_time::Array{Float32,2}
    forecast_time::Array{Float32,2}
end

function TrendBasis(degree_of_polynomial::Int, backcast_size::Int, forecast_size::Int)
    polynomial_size = degree_of_polynomial + 1
    backcast_time = hcat(
        [
            ((0:(backcast_size - 1)) / backcast_size) .^ i for i in 0:(degree_of_polynomial)
        ]...,
    )
    forecast_time = hcat(
        [
            ((0:(forecast_size - 1)) / forecast_size) .^ i for i in 0:(degree_of_polynomial)
        ]...,
    )

    return TrendBasis(polynomial_size, backcast_time', forecast_time')
end

function (basis::TrendBasis)(theta::AbstractArray)
    backcast = theta[:, 1:(basis.polynomial_size)] * basis.backcast_time
    forecast = theta[:, (end - basis.polynomial_size + 1):end] * basis.forecast_time
    return backcast, forecast
end

struct SeasonalityBasis{T<:AbstractArray}
    backcast_cos_template::T
    backcast_sin_template::T
    forecast_cos_template::T
    forecast_sin_template::T
end

function SeasonalityBasis(harmonics::Int, backcast_size::Int, forecast_size::Int)
    frequency = vcat(
        zeros(Float32, 1), collect(1:(harmonics / 2 * forecast_size)) / harmonics
    )

    backcast_grid = -2π * (collect(0:(backcast_size - 1)) / forecast_size) * frequency'
    forecast_grid = 2π * (collect(0:(forecast_size - 1)) / forecast_size) * frequency'

    backcast_cos_template = cos.(backcast_grid')
    backcast_sin_template = sin.(backcast_grid')
    forecast_cos_template = cos.(forecast_grid')
    forecast_sin_template = sin.(forecast_grid')

    return SeasonalityBasis(
        backcast_cos_template,
        backcast_sin_template,
        forecast_cos_template,
        forecast_sin_template,
    )
end

function (basis::SeasonalityBasis)(theta::AbstractArray)
    params_per_harmonic = size(theta, 2) ÷ 4
    backcast_harmonics_cos =
        theta[:, (2 * params_per_harmonic + 1):(3 * params_per_harmonic)] *
        basis.backcast_cos_template
    backcast_harmonics_sin =
        theta[:, (3 * params_per_harmonic + 1):end] * basis.backcast_sin_template
    backcast = backcast_harmonics_sin + backcast_harmonics_cos

    forecast_harmonics_cos = theta[:, 1:params_per_harmonic] * basis.forecast_cos_template
    forecast_harmonics_sin =
        theta[:, (params_per_harmonic + 1):(2 * params_per_harmonic)] *
        basis.forecast_sin_template
    forecast = forecast_harmonics_sin + forecast_harmonics_cos

    return backcast, forecast
end
