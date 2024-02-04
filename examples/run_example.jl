include("../src/AnisotropicDiffusionEstimate.jl")
using .AnisotropicDiffusionEstimate: generate_trajectory, EM_solve, loglike, loglikebased_Dθ_optim, getStandardDeviation


## Parameters
Dat = 10.0
Dbt = 5.0
Dθt = 1.0
dt = 0.01
ϵ = 0.01
N =  1000
seed = 1 
M = 10^4



## Generate sample trajectory
X, Y, θ = generate_trajectory(N, Dat, Dbt, Dθt, ϵ, dt, seed)


## Log likelihood of the initial parameters
ini_loglikelihood_value = loglike(X, Y, Dat, Dbt, Dθt, dt, ϵ, M)
println("Initial log likelihood: ", ini_loglikelihood_value)


## Apply EM algorithm
n_iteration = 20
Da_estimated, Db_estimated, Dθ_estimated = EM_solve(n_iteration, X, Y, 10.0, 2.0, 10.0, dt, ϵ, M)
println("Estimated values: ", Da_estimated, " ", Db_estimated, " ", Dθ_estimated)
## Note that especaially the convergence of Dθ is very slow.
## So after several iterations, moving to the next surrogate step is recommended.


## Apply surrogate based optimazation
search_range = (0.01, 10.0)
iterations = 15
num_start_samples = 15
Dθ_estimated = loglikebased_Dθ_optim(search_range, Da_estimated, Db_estimated, X, Y, dt, ϵ, M, iterations = iterations, num_start_samples = num_start_samples)
println("Estimated Dθ: ", Dθ_estimated)


## Calculate the standard deviation of the estimated parameters from Bipartite Gaussian Approximation based on surrogate model optimization
tar_diff = 2.0
iterations = 15
num_start_samples = 15


search_range_Da = (Da_estimated/5, Da_estimated*2)
Da_upper_std, Da_lower_std = getStandardDeviation("Da", search_range_Da, X, Y, Da_estimated, Db_estimated, Dθ_estimated, dt, ϵ, M, tar_diff, iterations = iterations, num_start_samples = num_start_samples)
println("Estimation result of Da : ", Da_estimated - Da_upper_std, " < Da < ", Da_estimated + Da_lower_std)


search_range_Db = (Db_estimated/5, Db_estimated*2)
Db_upper_std, Db_lower_std = getStandardDeviation("Db", search_range_Db, X, Y, Da_estimated, Db_estimated, Dθ_estimated, dt, ϵ, M, tar_diff, iterations = iterations, num_start_samples = num_start_samples)
println("Estimation result of Db : ", Db_estimated - Db_upper_std, " < Db < ", Db_estimated + Db_lower_std)


search_range_Dθ = (Dθ_estimated/5, Dθ_estimated*5)
Dθ_upper_std, Dθ_lower_std = getStandardDeviation("Dθ", search_range_Dθ, X, Y, Da_estimated, Db_estimated, Dθ_estimated, dt, ϵ, M, tar_diff, iterations = iterations, num_start_samples = num_start_samples)
println("Estimation result of Dθ : ", Dθ_estimated - Dθ_upper_std, " < Dθ < ", Dθ_estimated + Dθ_lower_std)


