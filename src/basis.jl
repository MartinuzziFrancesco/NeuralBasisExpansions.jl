function linear_space(backcast_length, forecast_length, is_forecast=true)
    horizon = is_forecast ? forecast_length : backcast_length
    return collect(range(1, stop=horizon, length=horizon) ./ horizon)
end

function generic_basis(t, thetas)
    lin = Dense(length(thetas), length(t))
    return lin(thetas)
end

function trend_basis(t, thetas)
    theta_size = size(thetas, 2)
    T = [t .^ i for i in 0:(theta_size-1)] # Create polynomial terms
    T_matrix = hcat(T...) # Each row is t.^i
    return T_matrix * transpose(thetas)
end

function seasonality_basis(t, thetas)
    theta_size = size(thetas, 2)
    p1, p2 = theta_size ÷ 2, theta_size ÷ 2 + (theta_size % 2)
    s1 = [cos.(2 * π * i .* t) for i in 1:p1]
    s2 = [sin.(2 * π * i .* t) for i in 1:p2]
    S_matrix = hcat(s1..., s2...) # Each row is a sinusoidal component
    return S_matrix * transpose(thetas)
end


struct BasisLayer
    fc_b
    fc_f
    basis_function_b
    basis_function_f
end

function BasisLayer(
    layer_size::Int,
    theta_size::Int,
    basis_function;
    backcast_length::Int=10,
    forecast_length::Int=5,
    share_thetas::Bool = false
)
    fc_b = Dense(layer_size, theta_size)
    fc_f = share_thetas ? fc_b : Dense(layer_size, theta_size)
    b_linspace = linear_space(backcast_length, forecast_length, false)
    f_linspace = linear_space(backcast_length, forecast_length, true)
    bf_b = basis_function $ b_linspace
    bf_f = basis_function $ f_linspace
    return BasisLayer(fc_b, fc_f, bf_b, bf_f)
end

function (bl::BasisLayer)(x)
    backcast = bl.basis_function_b(bl.fc_b(x))
    forecast = bl.basis_function_f(bl.fc_f(x))
    return backcast, forecast
end