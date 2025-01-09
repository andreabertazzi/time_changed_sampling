include("/Users/abertazzi1/Library/CloudStorage/Dropbox/PhD_TUDelft/Codes/Splitting schemes/algorithms.jl")
include("helper_tt.jl")

using Distributions
using ForwardDiff
using SpecialFunctions

# Setup target
dim = 2
μ = zeros(dim)
correlation = 0.
Σ = correlation*ones(dim,dim) + (1-correlation)I
Σ_inv = inv(Σ)
norm_Σ = norm(Σ)
target(x) = gamma((1+dim)/2)/(gamma(1/2)*π^(dim/2)*(norm_Σ^(1/2))*(1+transpose(x)*Σ_inv*x)^((1+dim)/2))
U(x)  = ((1+dim)/2) * log(1+transpose(x)*Σ_inv*x)
∇U(x) = 2*((1+dim)/2) * Σ_inv*x / (1+transpose(x)*Σ_inv*x)


s(x,α)  = exp(α*U(x))
∇U_s(x,α) = (1-α)*∇U(x)
unnorm_target_s(x,α) = exp(-(1-α)*U(x))
params = [0., 0.1, 0.2, 0.3, 0.4]
gradients = [x -> ∇U_s(x,p) for p in params]
targets = [x -> unnorm_target_s(x,p) for p in params]
speeds = [x -> s(x,p) for p in params]

δ = 0.2
batch_size = 1000
samplers = [(x,v) -> jump_metropolised_zzs(targets[i],gradients[i],δ,batch_size,x,v,speeds[i]) for i =1:length(params)]

threshold = 150
function test_fun(x::Vector{<:Real})
    # if all(x.>threshold)
    if norm(x)>threshold
        return 1
    else 
        return 0
    end
end

n_exp = 2500
n_batches = 1000
estimates = compare_samplers_parallel(samplers, n_exp, n_batches,test_fun,cond_init_zz)
# t_parallel = @elapsed(estimates = compare_samplers_parallel(samplers, n_exp, n_batches,test_fun,cond_init_zz))
# t_seq = @elapsed(estimates = compare_samplers(samplers, n_exp, n_batches,test_fun,cond_init_zz))

# now compute the means
mean_estimates = transpose(mean(estimates; dims = 2)[:,1,:])
var_estimates = transpose(var(estimates; dims = 2)[:,1,:])
truth = π * gamma((1+dim)/2)/(gamma(1/2)*π^(dim/2)) * (2/(dim-1)) * (1+threshold^2)^(-(dim-1)/2)
bias = mean((estimates[:,:,end].-truth);dims=2)
variance = var((estimates[:,:,end].-truth);dims=2)
mse = mean((estimates[:,:,end].-truth).^2;dims=2)

labs = ["a=$p" for p in params[1:4]]
labs = reshape(labs,1,:)
plot(1:batch_size:n_batches*batch_size, mean_estimates, lw=2, labels = labs,
    # yaxis=:log,
legend=:bottomright
        )
plot!(truth*ones(n_batches*batch_size),lc=:black)

errors = abs.(estimates.-truth)
mean_err = transpose(mean(errors; dims = 2)[:,1,:])
plot(1:batch_size:n_batches*batch_size, mean_err, lw=2, labels = labs,yaxis=:log,
    ylabel = "Mean absolute error", xlabel = "Iterations",
    # linecolor=:red, 
    ls=:auto, lc=:auto,
    xlims = [-1,1e6+50000],
    # ylims = [10^(-3),10^-2]
    )
# savefig(string("t-distribution_absoluteerror_thresh",threshold,".pdf"))

boxplot()
boxplot!(labs,transpose(estimates[:,:,end]), labels="",
    ylims=[-0.005,0.08], # color="white",
    )
hline!([truth],label="Ground truth",lc="red",lw=2,alpha=:0.5)

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


# a=0.1
# x_rng = range(-100, stop = 100, length = 1000)
# U(x)  = ((1+dim)/2) * log(1+x^2)
# c = 1
# b = 1
# r = 1
# # fn(x) = (c+exp((r*1+a)*U(x)))/(b*exp(a*U(x))+exp(r*U(x)))
# fn(x) = 
# U_new(x) = U(x) -log(fn(x))
# plot(x_rng,U_new.(x_rng),label="New",legend=:bottomright);
# # plot(x_rng,fn.(x_rng),label="New",legend=:bottomright);
# usual(x) = exp(a*U(x))
# U_usual(x) = U(x) -log(usual(x))
# plot!(x_rng,U_usual.(x_rng),label="Usual")

# target_new(x) = exp(-U(x)) * fn(x)
# plot(x,target_new.(x),label="New");
# target_old(x) = exp(-(1-a)*U(x))
# plot!(x,target_old.(x))