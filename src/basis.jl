function linear_space(backcast_length, forecast_length, is_forecast=true)
    horizon = is_forecast ? forecast_length : backcast_length
    return collect(range(1, stop=horizon, length=horizon) ./ horizon)
end

function generic_basis(t, thetas)
    lin = Dense(length(thetas), length(t)) ## to check!!
    return Dense(thetas)
end

function trend_basis(t, thetas)
    p = size(thetas, 2)
    @assert p <= 4 "thetas_dim is too big."
    T = reduce(hcat, [t.^i for i in 1:p])

    return thetas * transpose(T)
end

function seasonality_basis(t, thetas)
    p = size(thetas, 2)
    @assert p <= size(thetas, 1) "thetas_dim is too big."
    p1, p2 = p ÷ 2, p ÷ 2 + (p % 2)

    s1 = [cos.(2 * π * i .* t) for i in 1:p1]
    s2 = [sin.(2 * π * i .* t) for i in 1:p2]
    S = reduce(hcat, vcat(s1, s2))

    return thetas * transpose(S)
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
    backcast = basis_function_b(fc_b(x))
    forecast = basis_function_f(fc_f(x))
    return backcast, forecast
end