using Random, Distributions



"""
    generate_trajectory(N::Int, Da::Float64, Db::Float64, Dθ::Float64, ϵ::Float64, dt::Float64, seed::Int)

Create a trajectory for artificial anisotropic diffusion.

# input
- `N::Int`: The number of points.
- `Da::Float64`: The larger diffusion coefficient.
- `Db::Float64`: The smaller diffusion coefficient.
- `Dθ::Float64`: The rotational diffusion coefficient.
- `ϵ::Float64`: The noise intensity.
- `dt::Float64`: The time interval.
- `seed::Int`: The seed of random number generator.

# output
- `X::Array{Float64}`: The observed x-coordinate.
- `Y::Array{Float64}`: The observed y-coordinate.
- `θ::Array{Float64}`: The angle of trajectory.

Note that θ is not observed so we do not use it in the estimation.
"""

function generate_trajectory(N::Int, Da::Float64, Db::Float64, Dθ::Float64, ϵ::Float64, dt::Float64, seed::Int)
    Random.seed!(seed)

    x = zeros(N)
    y = zeros(N)
    θ = zeros(N)

    θ[1] = rand(Uniform(-pi/2, pi/2)) 
    Δθ = rand(Normal(0.0, sqrt(2.0 * Dθ * dt)), N-1)
    for i in 1:N-1
        θ[i+1] = θ[i] + Δθ[i]
    end

    Δx_tilda = rand(Normal(0.0 , sqrt(2.0 * Da * dt)) , N-1)
    Δy_tilda = rand(Normal(0.0 , sqrt(2.0 * Db * dt)) , N-1)

    Δx = @. Δx_tilda * cos(θ[1:end-1]) + Δy_tilda * (-sin(θ[1:end-1]))
    Δy = @. Δx_tilda * sin(θ[1:end-1]) + Δy_tilda * cos(θ[1:end-1])

    for i in 1:N-1 
        x[i+1] = x[i] + Δx[i]
        y[i+1] = y[i] + Δy[i]
    end

    X = x .+ rand(Normal(0.0, ϵ), N)
    Y = y .+ rand(Normal(0.0, ϵ), N)
    
    return X, Y, θ
end

