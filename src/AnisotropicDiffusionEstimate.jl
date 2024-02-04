module AnisotropicDiffusionEstimate

include("data_generation.jl")
include("estimation.jl")
include("loglikelihood.jl")
include("loglikebased_optim.jl")

export generate_trajectory, EM_solve, loglike, loglikebased_Dθ_optim, getStandardDeviation

end