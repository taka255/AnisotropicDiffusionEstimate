using SurrogateModelOptim

""" 
    loglikebased_Dθ_optim(search_range, iterations, num_start_samples, Da_est, Db_est, X, Y, dt, ϵ, M)

This function is used to optimize the log likelihood function using the surrogate model optimization method.
Note that the calculation of log_likelihood is much easier and accurate than EM_algorithm

# Arguments
- `search_range::Tuple{Float64, Float64}`: search range of Dθ
- `iterations::Int`: number of iterations
- `num_start_samples::Int`: number of start samples
- `Da_est::Float64`: the estimated diffusion coefficient of Da
- `Db_est::Float64`: the estimated diffusion coefficient of Db
- `X::Vector{Float64}`: x-coordinate of the trajectory
- `Y::Vector{Float64}`: y-coordinate of the trajectory
- `dt::Float64`: time step
- `ϵ::Float64`: standard deviation of the measurement noise
- `M::Int`: number of particles
"""


function loglikebased_Dθ_optim(search_range, Da_est, Db_est, X, Y, dt, ϵ, M; iterations = 15, num_start_samples = 15)
    f(x) = - loglike(X, Y, Da_est, Db_est, x[1],  dt, ϵ, M)
    opt = smoptimize(f, [search_range] ;
        options=SurrogateModelOptim.Options(
        iterations = iterations,
        num_start_samples = num_start_samples,
        smooth =:single,
        trace = :silent,
        ))

    return best_candidate(opt)[1,1]
end

"""
    loglike_parametric(X, Y, Da, Db, Dθ, dt, ϵ, M, parameter_value, parameter_name, tar)

This function is used to calculate the log likelihood of the trajectory for the given parameter value.
        
# Arguments
- `X::Vector{Float64}`: x-coordinate of the trajectory
- `Y::Vector{Float64}`: y-coordinate of the trajectory
- `Da::Float64`: Diffusion coefficient of the x-coordinate
- `Db::Float64`: Diffusion coefficient of the y-coordinate
- `Dθ::Float64`: Diffusion coefficient of the angle
- `dt::Float64`: Time step
- `ϵ::Float64`: Standard deviation of the measurement noise
- `M::Int`: Number of particles
- `parameter_value::Float64`: the value of the parameter
- `parameter_name::String`: name of the parameter (Da, Db, or Dθ)
- `tar::Float64`: target value of the log likelihood for Gaussian approximation
"""



function loglike_parametric(X, Y, Da, Db, Dθ, dt, ϵ, M, parameter_value, parameter_name, tar)
    if parameter_name == "Da"
        return (loglike(X, Y, parameter_value, Db, Dθ, dt, ϵ, M) - tar)^2
    elseif parameter_name == "Db"
        return (loglike(X, Y, Da, parameter_value, Dθ, dt, ϵ, M) - tar)^2
    elseif parameter_name == "Dθ"
        return (loglike(X, Y, Da, Db, parameter_value, dt, ϵ, M) - tar)^2
    else
        error("Invalid parameter name")
    end
end

"""
    getStandardDeviation(X, Y, Da_est, Db_est, Dθ_est, tar, dt, ϵ, M)

This function is used to calculate the standard deviation of the estimated parameters from Bipartite Gaussian Approximation.
    The "Bipartite Gaussian Approximation" method divides a function at its peak into left and right sections, 
    each approximated by a distinct Gaussian function around the peak.

# Arguments
- `parameter::String`: name of the parameter (Da, Db, or Dθ)
- `search_range::Tuple{Float64, Float64}`: search range of the parameter
- `X::Vector{Float64}`: x-coordinate of the trajectory
- `Y::Vector{Float64}`: y-coordinate of the trajectory
- `Da_est::Float64`: the estimated diffusion coefficient of Da
- `Db_est::Float64`: the estimated diffusion coefficient of Db
- `Dθ_est::Float64`: the estimated diffusion coefficient of Dθ
- `tar_diff::Float64`: target value of the log likelihood for Gaussian approximation
- `dt::Float64`: time step
- `ϵ::Float64`: standard deviation of the measurement noise
- `M::Int`: number of particles
- `tar_diff::Float64`: 
       This variable represents the constant subtracted from the maximum log-likelihood to assist in the Gaussian approximation process for variance calculation. 
       Altering this value modifies the reference point on the log-likelihood curve, with a subsequent scale transformation ensuring the precise calculation of sigma. 
   

# Optional arguments
- `iterations::Int=15`: number of iterations
- `num_start_samples::Int=15`: number of start samples

# Returns
- `Tuple{Float64, Float64}`: upper and lower standard deviation
"""

function getStandardDeviation(parameter::String, search_range, X, Y, Da_est, Db_est, Dθ_est, dt, ϵ, M, tar_diff; iterations = 15, num_start_samples = 15)
    max_log = loglike(X, Y, Da_est, Db_est, Dθ_est, dt, ϵ, M)
    tar = max_log - tar_diff


    f = (x) -> loglike_parametric(X, Y, Da_est, Db_est, Dθ_est, dt, ϵ, M, x[1], parameter, tar)
    

    est_value = if parameter == "Da"
        Da_est
    elseif parameter == "Db"
        Db_est
    elseif parameter == "Dθ"
        Dθ_est
    else
        error("Invalid parameter name")
    end
    
    search_range_r = [(est_value, search_range[2])]
    search_range_l = [(search_range[1], est_value)]

    opt1 = smoptimize(f, search_range_r; options=SurrogateModelOptim.Options(
            iterations = iterations,
            num_start_samples = num_start_samples,
            smooth =:single,
            trace = :silent
        ))

    opt2 = smoptimize(f, search_range_l; options=SurrogateModelOptim.Options(
            iterations = iterations,
            num_start_samples = num_start_samples,
            smooth =:single,
            trace = :silent
        ))

    upper_sigma = (best_candidate(opt1)[1,1] - est_value) / sqrt(2*tar_diff)
    lower_sigma = (est_value - best_candidate(opt2)[1,1]) / sqrt(2*tar_diff)

    return (upper_sigma, lower_sigma)
end
