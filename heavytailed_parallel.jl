include("helper_split.jl")
include("helper_tt.jl")
include("algorithms.jl")

using Distributions
using ForwardDiff, LinearAlgebra
using SpecialFunctions, ProgressBars

# Setup target
dim = 2
μ = zeros(dim)
Σ = diagm(ones(dim))
Σ_inv = diagm(ones(dim))
norm_Σ = norm(Σ)
target(x) = gamma((1+dim)/2)/(gamma(1/2)*π^(dim/2)*(norm_Σ^(1/2))*(1+transpose(x)*Σ_inv*x)^((1+dim)/2))
U(x)  = ((1+dim)/2) * log(1+transpose(x)*Σ_inv*x)
∇U(x) = 2*((1+dim)/2) * Σ_inv*x / (1+transpose(x)*Σ_inv*x)


s(x,α)  = exp(α*U(x))
∇U_s(x,α) = (1-α)*∇U(x)
unnorm_target_s(x,α) = exp(-(1-α)*U(x))
params = [0., 0.1, 0.2, 0.3]
gradients = [x -> ∇U_s(x,p) for p in params]
targets = [x -> unnorm_target_s(x,p) for p in params]
speeds = [x -> s(x,p) for p in params]

δ = 0.2
n_iter = 1000
samplers = [(x,v) -> jump_metropolised_zzs(targets[i],gradients[i],δ,n_iter,x,v,speeds[i]) for i =1:length(params)]

threshold = 150
function test_fun(x::Vector{<:Real})
    # if all(x.>threshold)
    if norm(x)>threshold
        return 1
    else 
        return 0
    end
end
truth = π * gamma((1+dim)/2)/(gamma(1/2)*π^(dim/2)) * (2/(dim-1)) * (1+threshold^2)^(-(dim-1)/2)


n_exp = 5000
n_batches = 1000
estimates = compare_samplers_parallel(samplers, n_exp, n_batches,test_fun,cond_init_zz)


labs = ["a=$p" for p in params[1:4]]
labs = reshape(labs,1,:)

errors = (estimates.-truth).^2 / truth^2
mean_err = transpose(median(errors; dims = 2)[:,1,:])
column_colors = [:red, :blue, :green, :orange]
# pl_mse = plot(ylabel = "Relative mean square error", xlabel = "Iterations",yaxis=:log,) 
pl_mse = plot(ylabel = "", xlabel = "",yaxis=:log, xlims=[0, n_batches*n_iter * 1.04]) 
for i in 1:4
    plot!(pl_mse, 1:n_iter:n_batches*n_iter, mean_err[:, i],lw=2, color=column_colors[i], label= labs[i])
end
display(pl_mse)
savefig(string("t-distribution_mse_thresh",threshold,".pdf"))


data = transpose(estimates[:, :, end])  

## Produce the boxplot
plot()
for i in 1:4
    boxplot!(fill(labs[i], size(data, 1)), data[:, i], 
        color=column_colors[i], 
        ylims=[-0.005,0.08], 
        # ylims=[-0.005,0.2], 
        markerstrokewidth=0,
        fillalpha=0.7, 
        markersize=:3,
        alpha=0.7,
        label="")
end
hline!([truth], label="Ground truth", lc=:red, lw=2, alpha=0.5)

# savefig(string("t-distribution_boxplot_thresh",threshold,".pdf"))


## Sava data as CSV file

# using CSV, DataFrames
# temp = dataset(DataFrame, "tips")

# names = ["alpha=$p" for p in params]
# df = DataFrame(
#     sampler = String[],
#     batch_nr = Int[],
#     estimate = Float64[],
# );
# function make_df!(df::DataFrame, values::Array{Float64},
#                     name_sampler::Vector{String})
#         for j in eachindex(name_sampler)
#             for i in eachindex(values[1,:,1])
#                 for k in eachindex(values[1,1,:])
#                     push!(
#                      df,
#                      Dict(
#                          :sampler => name_sampler[j],
#                          :batch_nr => k,
#                          :estimate => values[j,i,k],
#                          ),
#                      )
#                 end
#             end
#         end
# end

# make_df!(df,estimates,names)
# CSV.write(string("/Users/abertazzi1/Library/CloudStorage/Dropbox/PhD_TUDelft/Codes/Time transformations/t-distribution_estimates_threshold", threshold,".csv"),df)
