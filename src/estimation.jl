using Random, Distributions, LinearAlgebra, KernelDensity, StaticArrays

"""
    rempi(x::Float64)

Return the value of x in [-π/2, π/2].
"""
function rempi(x::Float64) 
    abs(x) <= pi/2 && return x

    n = rem2pi(x, RoundNearest)
    if -pi/2 < rem2pi(x, RoundNearest) < pi/2
        return n
    end
    if pi/2 <= rem2pi(x, RoundNearest) <= pi
        return n - pi
    end
    if -pi <= rem2pi(x, RoundNearest) <= -pi/2
        return n + pi
    end
end


"""
    my_cholesky!(A::AbstractMatrix)

Performs an in-place Cholesky decomposition on a 2x2 matrix.

# Arguments
- `A::AbstractMatrix`: A 2x2 matrix to be decomposed.

# Notes
The input matrix `A` is overwritten with its Cholesky factor.
"""
function my_cholesky!(A::AbstractMatrix)
    a, b, c, d = A[1, 1], A[1, 2], A[2, 1], A[2, 2]

    L11 = sqrt(a)
    L21 = b / L11
    L22 = sqrt(d - L21^2)

    A[1, 1] = L11
    A[1, 2] = 0.0
    A[2, 1] = L21
    A[2, 2] = L22
end


"""
    my_MvNormal!(mu::AbstractVector, sig::AbstractMatrix, tmp::AbstractVector)

Generates multivariate normal random numbers in-place without allocation, storing the result in `mu`.

# Arguments
- `mu::AbstractVector`: Mean vector where the generated random numbers will be added.
- `sig::AbstractMatrix`: Covariance matrix for the generation.
- `tmp::AbstractVector`: A temporary vector for holding standard normal variates.

# Notes
The function assumes `sig` has been Cholesky decomposed in advance.
"""
function my_MvNormal!(mu::AbstractVector, sig::AbstractMatrix, tmp::AbstractVector)
    randn!(tmp)
    my_cholesky!(sig)
    mu .+= sig * tmp
end



"""
    onestep_next_filter(X, Y, Da, Db, Dtheta, particle, M, tmpI, tmpF,  w, dt, ϵ)

Performs one step of the forward filtering process for the anisotropic diffusion model.

# Arguments
- `X::Float64`: The x-coordinate of the observed point.
- `Y::Float64`: The y-coordinate of the observed point.
- `Da::Float64`: The larger diffusion coefficient.
- `Db::Float64`: The smaller diffusion coefficient.
- `Dtheta::Float64`: The rotational diffusion coefficient.
- `particle::AbstractMatrix`: The particle matrix of size (3, M) where M is the number of particles.
- `M::Int`: The number of particles.
- `tmpI::AbstractVector`: A temporary vector for holding indices.
- `tmpF::AbstractVector`: A temporary vector for holding random numbers.
- `w::AbstractVector`: A vector for holding weights.
- `dt::Float64`: The time interval.
- `ϵ::Float64`: The noise intensity.

# Returns
- `particle::AbstractMatrix`: The updated particle matrix.
"""


function onestep_next_filter(X, Y, Da, Db, Dtheta, particle, M, tmpI, tmpF,  w, dt, ϵ) 
    Dbar = (Da + Db)*dt/2
    delD = (Da - Db)*dt

    theta = @view particle[1,:]
    x = @view particle[2,:]
    y = @view particle[3,:]

    mu1 = @MVector [X,Y]
    mu2 = @MVector zeros(Float32,2)
    sig2 = @MMatrix zeros(Float32,2,2)
    sig1 = @MMatrix [ϵ^2 0;0 ϵ^2]

    
    ## resampling
    for i in 1:M
        mu2[1] = x[i]
        mu2[2] = y[i]
        
        sig2[1,1] = 2*Dbar+delD*cos(2*theta[i])
        sig2[1,2] = delD*sin(2theta[i])
        sig2[2,1] = delD*sin(2theta[i])
        sig2[2,2] = 2Dbar-delD*cos(2theta[i])
        w[i] = 1/(2pi * sqrt(det(sig1+sig2))) * exp(- 1/2 * dot(mu1-mu2 , inv(sig1+sig2) * (mu1-mu2)))
    end
    
    w .= normalize(w,1)
    d = Categorical(w, check_args=false)
    s = sampler(d)  

    for j in eachindex(tmpI)  
        tmpI[j] = rand(s)  
    end

    @inbounds theta .= particle[1,tmpI]
    @inbounds x .= particle[2, tmpI]
    @inbounds y .= particle[3, tmpI] 

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
        particle[2,i], particle[3,i] = mu    
         
    end

    theta .+= rand!(Normal(0,sqrt(2Dtheta*dt)),tmpF)
    particle[1,:] = theta

    return particle 
end  

"""
    onestep_back_filter(X, Y, Da, Db, Dtheta, particle, M, tmpI, tmpF,  w, dt, ϵ)

Performs one step of the backward filtering process for the anisotropic diffusion model.

# Arguments
- `X::Float64`: The x-coordinate of the observed point.
- `Y::Float64`: The y-coordinate of the observed point.
- `Da::Float64`: The larger diffusion coefficient.
- `Db::Float64`: The smaller diffusion coefficient.
- `Dtheta::Float64`: The rotational diffusion coefficient.
- `particle::AbstractMatrix`: The particle matrix of size (3, M) where M is the number of particles.
- `M::Int`: The number of particles.
- `tmpI::AbstractVector`: A temporary vector for holding indices.
- `tmpF::AbstractVector`: A temporary vector for holding random numbers.
- `w::AbstractVector`: A vector for holding weights.
- `dt::Float64`: The time interval.
- `ϵ::Float64`: The noise intensity.

# Returns
- `particle::AbstractMatrix`: The updated particle matrix.
"""

function onestep_back_filter(X, Y, Da, Db, Dtheta, particle, M, tmpI, tmpF,  w,  dt, ϵ) 
    Dbar = (Da + Db)*dt/2
    delD = (Da - Db)*dt
    theta = @view particle[1,:]
    x = @view particle[2,:]
    y = @view particle[3,:]

    mu1 = @MVector [X,Y]
    mu2 = @MVector zeros(Float32,2)
    sig2 = @MMatrix zeros(Float32,2,2)

    ## update theta
    theta .+= rand!(Normal(0,sqrt(2Dtheta*dt)),tmpF)
    particle[1,:] = theta
    sig1 = @MMatrix [ϵ^2 0;0 ϵ^2]

    ## resampling
    for i in 1:M
        mu2[1] = x[i]
        mu2[2] = y[i]
        
        sig2[1,1] = 2*Dbar+delD*cos(2*theta[i])
        sig2[1,2] = delD*sin(2theta[i])
        sig2[2,1] = delD*sin(2theta[i])
        sig2[2,2] = 2Dbar-delD*cos(2theta[i])
        w[i] = 1/(2pi * sqrt(det(sig1+sig2))) * exp(- 1/2 * dot(mu1-mu2 , inv(sig1+sig2) * (mu1-mu2)))
    end

    normalize!(w,1)
    d = Categorical(w, check_args=false)
    s = sampler(d)  

    for j in eachindex(tmpI)  
        tmpI[j] = rand(s)  
    end

    @inbounds theta .= particle[1,tmpI]
    @inbounds x .= particle[2, tmpI]
    @inbounds y .= particle[3, tmpI]
    
    mu = @MVector zeros(Float64,2)
    sig = @MMatrix zeros(Float64,2,2)
    temp =  @MVector zeros(Float64,2)
    sig_tmp = similar(sig) 

    for i in 1:M  
        mu2[1] = x[i]
        mu2[2] = y[i]
        sig2[1,1] = 2*Dbar+delD*cos(2*theta[i])
        sig2[1,2] = delD*sin(2theta[i])
        sig2[2,1] = sig2[1,2]
        sig2[2,2] = 2Dbar-delD*cos(2theta[i])
        
        mul!(sig_tmp, sig1, inv(sig1+sig2))  
        mul!(sig, sig_tmp, sig2)

        mu .= sig*(sig1\mu1 + sig2\mu2)
        sig[2,1]=sig[1,2]
        my_MvNormal!(mu,sig,temp)
        particle[2,i], particle[3,i] = mu  
    end

    return particle 
end  

"""
    check_forward_message(X, Y, Da, Db, Dtheta, dt, ϵ, M)

Performs the forward message calculation.

# Arguments
- `X::Array{Float64}`: The x-coordinates of the observed points.
- `Y::Array{Float64}`: The y-coordinates of the observed points.
- `Da::Float64`: The larger diffusion coefficient.
- `Db::Float64`: The smaller diffusion coefficient.
- `Dtheta::Float64`: The rotational diffusion coefficient.
- `dt::Float64`: The time interval.
- `ϵ::Float64`: The noise intensity.
- `M::Int`: The number of particles.

# Returns
- `forward_result::Array{Float64,3}`: The result of the forward message calculation.
"""

function check_forward_message(X, Y, Da,Db,Dtheta, dt, ϵ, M)
    step_num = length(X)

    forward_result = zeros(step_num,3,M)

    # initial particle
    forward_result[1,1,:] = rand(Uniform(0,2pi),M)
    forward_result[1,2,:] = rand(Normal(X[1],ϵ),M)
    forward_result[1,3,:] = rand(Normal(Y[1],ϵ),M)

    # forward message as particles
    particle =  forward_result[1,:,:]
    w = zeros(Float32, M)
    tmpI = zeros(Int,M)
    tmpF = zeros(Float32,M)

    for t in 2:step_num 
        particle = onestep_next_filter(X[t], Y[t], Da, Db, Dtheta, particle, M, tmpI, tmpF, w, dt, ϵ)
        forward_result[t,:,:] = particle
    end

    return forward_result
end

"""
    check_backward_message(X, Y, Da, Db, Dtheta, dt, ϵ, M)

Performs the backward message calculation.

# Arguments
- `X::Array{Float64}`: The x-coordinates of the observed points.
- `Y::Array{Float64}`: The y-coordinates of the observed points.
- `Da::Float64`: The larger diffusion coefficient.
- `Db::Float64`: The smaller diffusion coefficient.
- `Dtheta::Float64`: The rotational diffusion coefficient.
- `dt::Float64`: The time interval.
- `ϵ::Float64`: The noise intensity.
- `M::Int`: The number of particles.

# Returns
- `backward_result::Array{Float64,3}`: The result of the backward message calculation.
"""


function check_backward_message(X::Array{Float64}, Y::Array{Float64}, Da::Float64, Db::Float64, Dtheta::Float64, dt::Float64, ϵ::Float64, M::Int)
    step_num = length(X)

    backward_result = zeros(step_num,3,M)

    # final particle
    backward_result[step_num,1,:] = rand(Uniform(0,2pi),M)
    backward_result[step_num,2,:] = rand(Normal(X[step_num],ϵ),M)
    backward_result[step_num,3,:] = rand(Normal(Y[step_num],ϵ),M)

    # backward message as particles
    particle =  backward_result[step_num,:,:]
    w = zeros(Float32, M)
    tmpI = zeros(Int,M)
    tmpF = zeros(Float32,M)

    for t in step_num-1:-1:1 
        particle = onestep_back_filter(X[t], Y[t], Da, Db, Dtheta, particle, M, tmpI, tmpF, w, dt, ϵ)
        backward_result[t,:,:] = particle
    end

    return backward_result
end

"""
    EM_onestep(X, Y, Da, Db, Dtheta, dt, ϵ, M)

Performs one step of the EM algorithm.

# Arguments
- `X::Array{Float64}`: The x-coordinates of the observed points.
- `Y::Array{Float64}`: The y-coordinates of the observed points.
- `Da::Float64`: The larger diffusion coefficient.
- `Db::Float64`: The smaller diffusion coefficient.
- `Dtheta::Float64`: The rotational diffusion coefficient.
- `dt::Float64`: The time interval.
- `ϵ::Float64`: The noise intensity.
- `M::Int`: The number of perticles.

# Returns
- `Da::Float64`: The updated larger diffusion coefficient.
- `Db::Float64`: The updated smaller diffusion coefficient.
- `Dtheta::Float64`: The updated rotational diffusion coefficient.
"""


function EM_onestep(X, Y, Da, Db, Dtheta, dt, ϵ, M)
    step_num = length(X)
    forward_result = zeros(step_num,3,M)
    backward_result = zeros(step_num,3,M)
    Threads.@threads for i in 1:2
        if i == 1
            forward_result .= check_forward_message(X, Y, Da,Db, Dtheta, dt, ϵ, M)
        elseif i == 2
            backward_result .= check_backward_message(X, Y, Da,Db, Dtheta, dt, ϵ, M);
        end
    end

    Da_av = zeros(step_num-2)
    Db_av = zeros(step_num-2)
    Dtheta_av = zeros(step_num-2)

    
    # calculate the joint distribution p(θ_t, θ_{t+1})
    Threads.@threads for t in 1:step_num-2 
        dx =  backward_result[t+1,2,:] - forward_result[t,2,:] 
        dy =  backward_result[t+1,3,:] - forward_result[t,3,:]
        theta = forward_result[t,1,:]
        w = @. -1/4 * (cos(theta)*dx+sin(theta)*dy)^2/(Da*dt) - 1/4*(sin(theta)*dx-cos(theta)*dy)^2/(Db*dt)
        offset = maximum(w)
        w = @. exp(w-offset)
        normalize!(w,1)

        new_theta =  theta .+ rand(Normal(0,sqrt(2Dtheta*dt)),M)
        w2 = pdf(kde(mod2pi.(backward_result[t+1,1,:])), mod2pi.(new_theta)) 
        normalize!(w2,1)

        Da_av[t] = sum(@. ((cos(theta)*dx + sin(theta)*dy)^2 * w) )
        Db_av[t] = sum(@. ((sin(theta)*dx - cos(theta)*dy)^2 * w) )
        Dtheta_av[t] = sum(@. (new_theta - theta)^2 * w2 )
    end

    len = length(Da_av)
    Da = sum(Da_av) / (2*(len)*dt)
    Db = sum(Db_av) / (2*(len)*dt)
    Dtheta = sum(Dtheta_av) / (2*(len)*dt)

    if(Da < Db)
        Da, Db = Db, Da
    end

    return Da, Db, Dtheta
end

"""
    EM_solve(ite , X, Y, Da0, Db0, Dtheta0, dt, ϵ, M)

Performs the EM algorithm.

# Arguments
- `ite::Int`: The number of iterations.
- `X::Array{Float64}`: The x-coordinates of the observed points.
- `Y::Array{Float64}`: The y-coordinates of the observed points.
- `Da0::Float64`: The initial larger diffusion coefficient.
- `Db0::Float64`: The initial smaller diffusion coefficient.
- `Dtheta0::Float64`: The initial rotational diffusion coefficient.
- `dt::Float64`: The time interval.
- `ϵ::Float64`: The noise intensity.
- `M::Int`: The number of particles.

# Returns
- `Da::Float64`: The estimated larger diffusion coefficient.
- `Db::Float64`: The estimated smaller diffusion coefficient.
- `Dtheta::Float64`: The estimated rotational diffusion coefficient.
"""


function EM_solve(ite::Int, X::Array{Float64}, Y::Array{Float64}, Da0::Float64, Db0::Float64, Dtheta0::Float64, dt::Float64, ϵ::Float64, M::Int)
    Da_ite = zeros(ite+1)
    Db_ite = zeros(ite+1)
    Dtheta_ite = zeros(ite+1)
    
    Da = Da0
    Db = Db0
    Dtheta = Dtheta0
    Da_ite[1]=Da
    Db_ite[1]=Db
    Dtheta_ite[1]=Dtheta

    for i in 1:ite
        Da, Db, Dtheta = EM_onestep(X, Y, Da, Db, Dtheta, dt, ϵ, M)  
        Da_ite[i+1]=Da
        Db_ite[i+1]=Db
        Dtheta_ite[i+1]=Dtheta
    end

    return Da_ite[end], Db_ite[end], Dtheta_ite[end] 
end