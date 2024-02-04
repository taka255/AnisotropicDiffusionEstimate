"""
    one_step_next_filter_like!(X, Y, Da, Db, Dtheta, theta, x, y, M, tmpI, tmpF,  w, dt, ϵ)

Calculate the log likelihood of the next step of the particle filter.

# Arguments
- `X::Float64`: x-coordinate of the next step
- `Y::Float64`: y-coordinate of the next step
- `Da::Float64`: Diffusion coefficient of the x-coordinate
- `Db::Float64`: Diffusion coefficient of the y-coordinate
- `Dtheta::Float64`: Diffusion coefficient of the angle
- `theta::Array{Float64,1}`: Angle of the particles
- `x::Array{Float64,1}`: x-coordinate of the particles
- `y::Array{Float64,1}`: y-coordinate of the particles
- `M::Int`: Number of particles
- `tmpI::Array{Int,1}`: Temporary array for resampling
- `tmpF::Array{Float64,1}`: Temporary array for resampling

# Returns

- `log_like::Float64`: Log likelihood of the next step
"""


function onestep_next_filter_like!(X::Float64, Y::Float64, Da::Float64, Db::Float64, Dtheta::Float64, theta::Array{Float64,1}, x::Array{Float64,1}, y::Array{Float64,1}, M::Int, tmpI::Array{Int,1}, tmpF::Array{Float64,1}, w::Array{Float64,1}, dt::Float64, ϵ::Float64)
    Dbar = (Da + Db)*dt/2
    delD = (Da - Db)*dt

    mu1 = @MVector [X,Y]
    mu2 = @MVector zeros(Float64,2)
    sig1 = @MMatrix [ϵ^2 0;0 ϵ^2]
    sig2 = @MMatrix zeros(Float64,2,2)
    

    s = 1/(2pi * sqrt(ϵ^4+4*Dbar*ϵ^2+4*Da*Db*dt*dt))
    ## resampling
    for i in 1:M
        mu2[1] = x[i]
        mu2[2] = y[i]
        
        sig2[1,1] = 2*Dbar+delD*cos(2*theta[i])
        sig2[1,2] = delD*sin(2theta[i])
        sig2[2,1] = delD*sin(2theta[i])
        sig2[2,2] = 2Dbar-delD*cos(2theta[i])

        # Calculating the constant C
        w[i] = exp(- 1/2 * dot(mu1-mu2 , inv(sig1+sig2) * (mu1-mu2)))
    end

    log_like = log(mean(w)) + log(s)
    normalize!(w,1)
    d = Categorical(w, check_args=false)
    s = sampler(d)    
    for j in 1:M 
        tmpI[j] = rand(s) 
    end
    
    for i in 1:M
        tmpF[i] = theta[tmpI[i]]
    end
    theta .= tmpF
    for i in 1:M
        tmpF[i] = x[tmpI[i]]
    end
    x .= tmpF
    for i in 1:M
        tmpF[i] = y[tmpI[i]]
    end
    y .= tmpF
    

    temp = @MVector zeros(Float64,2)
    mu = @MVector zeros(Float64,2)
    sig = @MMatrix zeros(Float64,2,2)
    tmp = similar(sig) 

    for i in 1:M
        mu2[1] = x[i]
        mu2[2] = y[i]
        sig2[1,1] = 2*Dbar+delD*cos(2*theta[i])
        sig2[1,2] = delD*sin(2theta[i])
        sig2[2,1] = sig2[1,2]
        sig2[2,2] = 2Dbar-delD*cos(2theta[i])
        
        mul!(tmp, sig1, inv(sig1+sig2))  
        mul!(sig, tmp, sig2)
        
        mu .= sig*(sig1\mu1 + sig2\mu2)
        sig[2,1]=sig[1,2]
        my_MvNormal!(mu,sig,temp)
        x[i], y[i] = mu    
         
    end

    theta .+= rand!(Normal(0,sqrt(2Dtheta*dt)),tmpF)

    return log_like
end  

"""
    loglike(X, Y, Da, Db, Dtheta, dt, ϵ, M)

Calculate the log likelihood of the trajectory.

# Arguments
- `X::Vector{Float64}`: x-coordinate of the trajectory
- `Y::Vector{Float64}`: y-coordinate of the trajectory
- `Da::Float64`: Diffusion coefficient of the x-coordinate
- `Db::Float64`: Diffusion coefficient of the y-coordinate
- `Dtheta::Float64`: Diffusion coefficient of the angle
- `dt::Float64`: Time step
- `ϵ::Float64`: Standard deviation of the measurement noise
- `M::Int`: Number of particles

# Returns

- `log_like_sum::Float64`: Log likelihood of the trajectory
"""

function loglike(X, Y, Da, Db, Dtheta, dt, ϵ, M)
    theta = rand(Uniform(0,2pi),M)
    x = rand(Normal(X[1],ϵ),M)
    y = rand(Normal(Y[1],ϵ),M)

    w = zeros(Float64, M)
    tmpI = zeros(Int,M)
    tmpF = zeros(Float64,M)
    log_like = 0

    log_like_sum = 0

    N = length(X)
    for t in 2:N
        log_like = onestep_next_filter_like!(X[t], Y[t], Da, Db, Dtheta, theta,x,y , M, tmpI, tmpF, w, dt, ϵ)
        log_like_sum += log_like
    end

    return log_like_sum
end
