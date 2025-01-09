using Statistics, LinearAlgebra, Compat, Plots, LaTeXStrings, StatsBase, StatsPlots
using BenchmarkTools, Optim

struct skeleton
  position::Array{Float64,1}
  velocity::Array{Float64,1}
  time::Float64
end

function getPosition(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0, want_array::Bool=false)
  if i_end == 0
    i_end = length(skele)
  end
  n_samples = i_end - i_start + 1
  dim = size(skele[1].position,1)
  if want_array
    position = Array{Float64,2}(undef,dim,n_samples)
    for i = i_start:i_end
      position[:,i] = skele[i].position
    end
  else
    position = Vector{Vector{Float64}}(undef,0)
    for i = i_start:i_end
      push!(position,skele[i].position)
    end
  end
  return position
end

function draw_exponential_time(λ::Real)
  if λ <= zero(λ)  # zero of same type as λ
    return Inf
  else
    U = rand()
    return -log(U) / λ # don't use for couplings ;)
  end
end

function flip_v(vel::Int64, δ::Float64, λ::Real)
    proposed_time = draw_exponential_time(λ)
    if proposed_time <= δ
        vel = -vel
    end
    vel
end

function flip!(vel::Vector, i::Integer)
  temp = vel[i]
  vel[i] = -temp
end

function reflect(gradient::Vector{Float64}, v::Vector{Float64})
  return v - 2 * (transpose(gradient) * v / dot(gradient,gradient)) * gradient
end

function define_gradient_gaussian(dim::Int64)
  Σ = ρ * ones(dim, dim) + (1 - ρ)I
  Σ_inv = inv(Σ)
  μ = zeros(dim)
  ∇U(x) = Σ_inv * (x - μ)
  return ∇U
end

function define_gradient_gaussian_variance(dim::Int64,variance::Real)
  Σ = variance.*diagm(ones(dim))
  Σ_inv = inv(Σ)
  μ = zeros(dim)
  ∇U(x) = Σ_inv * (x - μ)
  return ∇U
end

function define_gradient_gaussian_correlation(dim::Int64, ρ::Real)
  Σ = ρ * ones(dim, dim) + (1 - ρ)I
  Σ_inv = inv(Σ)
  μ = zeros(dim)
  ∇U(x) = Σ_inv * (x - μ)
  return ∇U
end


function switchingtime(a::Real, b::Real, u::Real=rand())
# generate switching time for rate of the form max(0, a + b s)
  if (b > 0)
    if (a < 0)
      return -a/b + switchingtime(0.0, b, u);
    else # a >= 0
      return -a/b + sqrt(a^2/b^2 - 2 * log(1-u)/b);
    end
  elseif (b == 0) # degenerate case
    if (a < 0)
      return Inf;
    else # a >= 0
      return -log(1-u)/a;
    end
  else # b <= 0
    if (a <= 0)
      return Inf;
    else # a > 0
      y = -log(1-u); t1=-a/b;
      if (y >= a * t1 + b *t1^2/2)
        return Inf;
      else
        return -a/b - sqrt(a^2/b^2 + 2 * y /b);
      end
    end
  end
end

function discretise(skel_chain::Vector{skeleton}, Δt::Float64, t_fin::Real)
  # Discretises the process described in skel_chain with step Δt for the interval
  # starting with the first time in skel_chain up to time t_fin (or the nearest
  # time point in the grid). The initial position is the first position in skel_chain.
  # The output is the process at discrete times, where the
  # first point in skel_chain is not considered.
  time = skel_chain[1].time
  i_skel = 1
  dim_skel = length(skel_chain)
  dim_temp = Int(round((t_fin-time)/Δt))
  d = length(skel_chain[1].position)
  temp = Array{Float64,2}(undef,d, dim_temp+1)
  temp[:,1] = skel_chain[1].position #this must be at time 0.0
  for k = 1:dim_temp
    time += Δt
    if i_skel==dim_skel || time <= skel_chain[i_skel+1].time
      temp[:,k+1]= temp[:,k] + skel_chain[i_skel].velocity*Δt;
    else
      while ((i_skel+1) <= dim_skel) && (time > skel_chain[i_skel+1].time)
        i_skel+=1;
      end
      t_left = time - skel_chain[i_skel].time;
      if (t_left > Δt)
        temp[:,k+1]= temp[:,k] + skel_chain[i_skel].velocity*Δt;
      else
        temp[:,k+1]= skel_chain[i_skel].position + skel_chain[i_skel].velocity*t_left;
      end
    end
  end
  temp
end

function estimate_inv_meas(positions::Vector{Float64}, bin_size::Float64, limits::Real; band_avg::Int64 = 3)
    nr_bins = Int64(abs(limits)/bin_size)  # actually half nr bins
    grid_pts = [i*bin_size for i=-nr_bins:nr_bins] # hence length(grid_pts) = 2nr_bins+1
    count_pts = (count_pts_bins(positions,grid_pts))/bin_size
    interpolate_and_average(count_pts,grid_pts,bin_size;band_avg=band_avg)
end

function count_pts_bins(positions::Vector{Float64},grid_pts::Vector{Float64};
                        normalise::Bool = true)
  nr_bins = length(grid_pts)-1
  count_pts = Vector{Float64}(zeros(nr_bins))
  nr_pts = length(positions)
  for i=1:(nr_bins)
    for j=1:nr_pts
      if (positions[j] <= grid_pts[i+1]) && (positions[j] > grid_pts[i])
        count_pts[i]+=1
      end
    end
  end
  if normalise
    pts_considered = sum(count_pts) # some points might be out of the bins region
    count_pts = count_pts/pts_considered
  end
  count_pts
end

function interpolate_and_average(count_pts::Vector{Float64}, grid_pts::Vector{Float64}, bin_size::Float64; band_avg::Int64 = 3)
    nr_pts = length(grid_pts)
    inv_meas_vec = Vector{Function}(undef,nr_pts-1)
    avg_neighbours = [avg_neigh(count_pts, i, band_avg) for i = 1:(nr_pts)]
    # avg_neighbours = avg_neighbours/sum(avg_neighbours)
    slope = [(avg_neighbours[i+1]-avg_neighbours[i])/bin_size for i = 1:(nr_pts-1)]
    grid_pts = grid_pts .+ bin_size/2
    for i = 1:(nr_pts-1)
      # avg_neighbours = avg_neigh(count_pts, i, band_avg)
      # slope = (avg_neighbours[i+1]-avg_neighbours[i])/bin_size
      # inv_meas[i] = inv_meas(x) + (slope*(x-grid_pts[i]) + avg_neighbours[i])*indicator_fn(x,grid_pts[i],grid_pts[i+1])
      func(x) = (slope[i]*(x-grid_pts[i]) + avg_neighbours[i])
      inv_meas_vec[i] = func
      # inv_meas[i] = @eval(x) (slope[i]*(x-grid_pts[i]) + avg_neighbours[i])
    end
    return (inv_meas_vec,grid_pts)
end

function avg_neigh(points::Vector{Float64}, ind::Int64, band::Int64)
    avg = 0
    for i = -band:band
      if (ind+i >= 1) && (ind+i<=length(points))
          avg += points[ind+i]
      end
    end
    return avg/(2*band+1)
end

function piecewise(x::Float64, breakpts::Vector{Float64}, f::Vector{Function})
       @assert(issorted(breakpts))
       @assert(length(breakpts) == length(f)+1)
       b = searchsortedfirst(breakpts, x)
       return f[b](x)
end

function indicator_fn(x::Real, a::Real, b::Real)
    if x<=b && x>=a
      1
    else
      0
    end
end

function compute_tv_dist(fn::Function, roots::Vector{Float64}, probdens::Function)
  sort!(roots)  #makes sure sorted from smallest to largest
  signs = Vector{Int}(undef,0)
  assign_signs!(signs,fn,roots)
  pushfirst!(roots, -Inf)
  push!(roots,Inf)
  integral_pos = 0
  for i = 1 : length(roots)-1
    if signs[i] > 0
      integ, err = quadgk(x -> (fn(x)*probdens(x)), roots[i], roots[i+1], rtol=1e-8)
      integral_pos += integ
    end
  end
  integral_pos
end

function assign_signs!(signs::Vector{Int}, fn::Function, roots::Vector{Float64})
  decide_sign!(signs,fn,-2*abs(roots[1]))
  # if roots[1] < 0
  #   decide_sign!(signs,fn,2*roots[1])
  # else
  #   decide_sign!(signs,fn,roots[1]/2)
  # end
  for i = 1 : length(roots)-1
    decide_sign!(signs,fn,(roots[i]+roots[i+1])/2)
  end
  decide_sign!(signs,fn,2*abs(roots[end]))
end

function decide_sign!(signs::Vector{Int},fn::Function,val::Real)
  if fn(val) < 0
    push!(signs,-1)
  else
    push!(signs,1)
  end
end
# function add_entries_errors!(df::DataFrame, errs::Array{Float64},
#                     name_sampler::Vector{String}, rr::LinRange{Float64})
#           for j = 1 : length(name_sampler)
#             for i = 1 : length(rr)
#             push!(
#                   df,
#                   Dict(
#                       :sampler => name_sampler[j],
#                       :refresh_rate => rr[i],
#                       :error_radius => errs[i,j]
#                       ),
#                   )
#             end
#           end
# end

function create_matrix_errors!(avg_err::Array{Float64,2}, errors::Array{Float64}, n_samplers::Int64)
    for i = 1 : n_samplers
      avg_err[:,i] = transpose(mean(errors[i,:,:,1]; dims=1))
    end
end

function string_names_samplers!(names_samplers::Vector{String})
    names_samplers[1] = "Splitting BRDRB"
    names_samplers[2] = "Splitting DRBRD"
    names_samplers[3] = "Splitting BDRDB"
    names_samplers[4] = "Splitting RDBDR"
    names_samplers[5] = "Splitting DBRBD"
    names_samplers[6] = "Splitting RBDBR"
    names_samplers[7] = "Splitting DBD"
    names_samplers[8] = "Splitting BDB"
    names_samplers[9] = "Splitting DR_B_DR"
    names_samplers[10] ="Splitting B_DR_B"
end

function compute_radius(covar_matrix::Function, corr::Vector{Float64}, dim::Vector{Int64})
    truth = Array{Float64,2}(undef,length(corr),length(dim))
    for i = 1 : length(corr)
        for j = 1 : length(dim)
            truth[i,j] = Float64( sum( diag( covar_matrix(corr[i],dim[j]) ) ) )
        end
    end
    truth
end



## for future work
function simulate_and_thin(sampler::Function,param::Real, cond_init::Vector{<:Real}, iterations_per_batch::Integer, n_thinned::Integer)
  z = cond_init
  chain = Vector{<:Real}(undef,n_thinned)
  for j = 1 : n_thinned
    for n = 1 : iterations_per_batch
        z = sampler(z, param)
    end
    chain[j] = copy(z)
    print("Simulation progress: ", floor(j/n_thinned*100), "% \r")
  end
  chain
end