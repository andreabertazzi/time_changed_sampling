## Auxiliary functions


struct skeleton_speed
  position::Vector{<:Real}
  speed::Real
  time::Real
end

struct skeleton_pos
  position::Vector{<:Float64}
  time::Float64
end

function discretise_from_skeleton(skele::Vector{skeleton},delta::Real)

  discr = Vector{typeof(skele[1].position)}()
  x = copy(skele[1].position)
  v = copy(skele[1].velocity)
  cur_time = 0.
  push!(discr, copy(x))
  t_left = delta

  for (_,sk) in enumerate(skele[2:end])

    t_to_next_jump = sk.time - cur_time

    while t_to_next_jump > t_left
      x += v * t_left
      cur_time += t_left
      push!(discr, copy(x))
      t_to_next_jump -= t_left
      t_left = delta
    end

    x = copy(sk.position)
    v = copy(sk.velocity)
    cur_time = sk.time
    t_left -= t_to_next_jump

  end

  discr

end



function approximate_timechanged_skele(skele::Vector, speed::Function, delta::Real)

  discr = Vector{skeleton_speed}()
  x = skele[1]
  cur_time = 0.
  cur_speed = speed(x)
  push!(discr, skeleton_speed(x,cur_speed,0.))
  cur_time = delta / cur_speed

  for (_,sk) in enumerate(skele[2:end])

    x = copy(sk)
    cur_speed = speed(x)
    push!(discr, skeleton_speed(x,cur_speed,cur_time))
    cur_time += delta / cur_speed

  end

  discr

end

function approximate_timechanged_skele(skele::Vector{skeleton}, speed::Function, delta::Real)

  discr = Vector{skeleton}()
  x = copy(skele[1].position)
  v = copy(skele[1].velocity)
  cur_speed = speed(x)
  cur_time = 0.
  new_time = 0.
  push!(discr, skeleton(copy(x),copy(v),new_time))
  t_left = delta

  for (_,sk) in enumerate(skele)

    t_to_next_jump = sk.time - cur_time

    while t_to_next_jump > t_left
      x += v * t_left
      cur_time += t_left
      new_time += t_left / cur_speed
      cur_speed = speed(x)
      push!(discr, skeleton(copy(x),copy(v),new_time))
      t_to_next_jump -= t_left
      t_left = delta
    end

    x = copy(sk.position)
    v = copy(sk.velocity)
    cur_time = sk.time
    new_time += t_to_next_jump / cur_speed
    cur_speed = speed(x)
    push!(discr, skeleton(copy(x),copy(v),new_time))
    t_left -= t_to_next_jump

  end

  discr

end


function make_into_jumpproc(skel::Vector{skeleton}, speed::Function; seq_E::Vector{Float64} = [.0])

  jump_skel = skeleton_pos[]
  if seq_E == [.0]
    seq_E = -log.(rand(length(chain)))
  end
  t = 0.0
  for (iii,sk) in enumerate(skel)
    push!(jump_skel,skeleton_pos(copy(sk.position),t))
    t += seq_E[iii] / speed(sk.position)
  end
  jump_skel

end


function make_into_jumpproc(chain::Vector{Vector{Float64}}, speed::Function; seq_E::Vector{Float64} = [.0])

  jump_skel = skeleton_pos[]
  if seq_E == [.0]
    seq_E = -log.(rand(length(chain)))
  end
  t = 0.0
  for (iii,ch) in enumerate(chain)
    push!(jump_skel,skeleton_pos(copy(ch),t))
    t += seq_E[iii] / speed(ch)
  end
  jump_skel

end

# function make_into_jumpproc(chain::Vector{Vector{Float64}}, speed::Function; seq_E::Vector{Float64} = [.0])
  # n_jumps = length(chain)
  # jump_skel = skeleton_pos[]
  # if seq_E == [.0]
  #   seq_E = -log.(rand(n_jumps))
  # end
  # t = 0.0
  # push!(jump_skel,skeleton_pos(chain[1],t))
  # for iii = 2 : n_jumps
  #   t += seq_E[iii-1] / speed(chain[iii-1])
  #   push!(jump_skel,skeleton_pos(copy(chain[iii]),t))
  # end
  # jump_skel
# end


function discretise_jumpprocess(skel_chain::Vector{skeleton}, delta::Real)
  discr = Vector{typeof(skel_chain[1].position)}()
  x = copy(skel_chain[1].position)
  cur_time = 0.
  push!(discr, copy(x))
  t_left = delta

  for (_,sk) in enumerate(skel_chain[2:end])

    t_to_next_jump = sk.time - cur_time

    while t_to_next_jump > t_left
      cur_time += t_left
      push!(discr, copy(x))
      t_to_next_jump -= t_left
      t_left = delta
    end

    x = copy(sk.position)
    cur_time = sk.time
    t_left -= t_to_next_jump

  end

  discr
end

function discretise_jumpprocess(skel_chain::Vector{skeleton_pos}, delta::Real)

  discr = Vector{typeof(skel_chain[1].position)}()
  x = copy(skel_chain[1].position)
  cur_time = 0.
  push!(discr, copy(x))
  t_left = delta

  for (_,sk) in enumerate(skel_chain[2:end])

    t_to_next_jump = sk.time - cur_time

    while t_to_next_jump > t_left
      cur_time += t_left
      push!(discr, copy(x))
      t_to_next_jump -= t_left
      t_left = delta
    end

    x = copy(sk.position)
    cur_time = sk.time
    t_left -= t_to_next_jump

  end

  discr

end


function define_gaussian_mixture(μ_1,μ_2,Σ_inv,λ)
    w = (μ_2+μ_1)/2
    v = (μ_2-μ_1)/2
    α = λ *exp(-0.5*transpose(w-μ_1)*Σ_inv*(w-μ_1)) * exp(transpose(w-μ_1)*Σ_inv*w)
    β = (1-λ) *exp(-0.5*transpose(w-μ_2)*Σ_inv*(w-μ_2)) * exp(transpose(w-μ_2)*Σ_inv*w)
    U_1(x) = 0.5*transpose(x-w)*Σ_inv*(x-w)
    U_2(x) = -log(α*exp(-transpose(Σ_inv*v)*x) + β*exp(transpose(Σ_inv*v)*x))
    U(x) = U_1(x) + U_2(x)
    ∇U_1(x) = Σ_inv*(x-w)
    m(x) = transpose(Σ_inv*v)*x
    ∇U_2(x) = (Σ_inv*v) * (α*exp(-m(x))-β*exp(m(x)))/(α*exp(-m(x))+β*exp(m(x)))
    ∇U(x) = ∇U_1(x) + ∇U_2(x)
    return (U,∇U)
end

function gaussian_pdf(μ::Vector{Float64},Σ::Matrix{Float64}, Σ_inv::Matrix{Float64})
    fun(x) = 1/((2*π)^(length(μ)/2) * (det(Σ))^(1/2)) * exp(-0.5*dot((x-μ),Σ_inv*(x-μ)))
    fun
end

function gaussian_pdf(μ::Float64,Σ::Float64, Σ_inv::Float64)
    fun(x) = 1/(2*π*Σ)^(1/2) * exp(-0.5*Σ_inv*(x-μ)^2)
    fun
end

function estimate_expectation(sample::Vector{skeleton}, observable::Function)
    # estimate expectation using the jump process
    # d_func = length(observable)
    n_samples = length(sample)
    cur = zeros(n_samples-1)
    cur[1] = observable(sample[1].position)
    for i in eachindex(sample[1:end-2])
      cur[i+1] = cur[i] + 
                  (sample[i+1].time - sample[i].time)/sample[i+1].time * (observable(sample[i+1].position)-cur[i])
    end
    cur 
end

function compare_samplers(samplers::Vector{<:Function},
                            n_exp::Int64,
                            n_batches::Int64,
                            t_func::Function,
                            init_cond::Function)
  
      estimates = zeros(length(samplers),n_exp,n_batches)
      for n = 1:n_exp
        println("Starting experiment $n")
        (x,v) = init_cond()
        for (i,sampler) in enumerate(samplers)
          (x_init,v_init) = (x,v)
          temp = 0
          t_horizon = 0 
          for j = 1:n_batches
            sk_chain = sampler(x_init,v_init)
            t_horizon += sk_chain[end].time
            update_est = sum([t_func(sk_chain[i].position) * (sk_chain[i+1].time-sk_chain[i].time) for i=1:length(sk_chain)-1])
            temp = temp + (1/t_horizon) * (update_est - (sk_chain[end].time) * temp) 
            estimates[i,n,j] = temp
            (x_init,v_init) = (sk_chain[end].position,round.(Int,sk_chain[end].velocity))
          end
        end
      end
      estimates
end

function compare_samplers(samplers::Vector{<:Function},
                            n_exp::Int64,
                            n_batches::Int64,
                            t_func::Function,
                            init_cond::Function)
  
      estimates = zeros(length(samplers),n_exp,n_batches)
      for n = 1:n_exp
        println("Starting experiment $n")
        (x,v) = init_cond()
        for (i,sampler) in enumerate(samplers)
          (x_init,v_init) = (x,v)
          temp = 0
          t_horizon = 0 
          for j = 1:n_batches
            sk_chain = sampler(x_init,v_init)
            t_horizon += sk_chain[end].time
            update_est = sum([t_func(sk_chain[i].position) * (sk_chain[i+1].time-sk_chain[i].time) for i=1:length(sk_chain)-1])
            temp = temp + (1/t_horizon) * (update_est - (sk_chain[end].time) * temp) 
            estimates[i,n,j] = temp
            (x_init,v_init) = (sk_chain[end].position,round.(Int,sk_chain[end].velocity))
          end
        end
      end
      estimates
end

function compare_samplers_parallel(samplers::Vector{<:Function},
  n_exp::Int64,
  n_batches::Int64,
  t_func::Function,
  init_cond::Function)

  estimates = zeros(length(samplers),n_exp,n_batches)
  Threads.@threads for n = 1:n_exp
    println("Starting experiment $n")
    (x,v) = init_cond()
    for (i,sampler) in enumerate(samplers)
      (x_init,v_init) = (x,v)
      temp = 0
      t_horizon = 0 
      for j = 1:n_batches
        sk_chain = sampler(x_init,v_init)
        t_horizon += sk_chain[end].time
        update_est = sum([t_func(sk_chain[i].position) * (sk_chain[i+1].time-sk_chain[i].time) for i=1:length(sk_chain)-1])
        temp = temp + (1/t_horizon) * (update_est - (sk_chain[end].time) * temp) 
        estimates[i,n,j] = temp
        (x_init,v_init) = (sk_chain[end].position,round.(Int,sk_chain[end].velocity))
      end
    end
  end
  estimates
end

function jump_metropolised_zzs(target::Function,
        ∇U::Function,
        avgstepsize::Float64,
        N::Integer,
        x_init::Vector{Float64},
        v_init::Vector{Int64},
        s::Function,
        # seq_exp::Vector{Float64}
        )
    sk = zzs_metropolis_randstepsize(target,∇U,avgstepsize,N,x_init,v_init)
    make_into_jumpproc(sk, s)
  end

function zzs_jump(∇U::Function,
    stepsize::Float64,
    N::Integer,
    x_init::Vector{Float64},
    v_init::Vector{<:Real},
    s::Function,
    # seq_exp::Vector{Float64}
    )
sk = splitting_zzs_DBD(∇U,stepsize,N,x_init,v_init)
make_into_jumpproc(sk, s)
end

function cond_init_zz()
    (randn(dim),rand((-1,1),dim))
end

function ULA_jump(∇U::Function,
  δ::Float64,
  N::Integer,
  x_init::Vector{Float64},
  s::Function,
  # seq_exp::Vector{Float64}
  )
    sk = ULA_sim(∇U,δ,N,x_init)
    make_into_jumpproc(sk, s)
end


# function compare_samplers_parallel_imag(samplers::Vector{<:Function},
#   n_batches::Int64,
#   t_func::Function,
#   init_cond::Function)

#   estimates = zeros(length(samplers),n_batches)
#   Threads.@threads for (i,sampler) in collect(enumerate(samplers))
#     # println("Starting sampler $i")
#     x_init = init_cond()
#     # temp = zeros(length(x_init))
#     temp = 0
#     t_horizon = 0 
#     for j = 1:n_batches
#         sk_chain = sampler(x_init)
#         t_horizon += sk_chain[end].time
#         update_est = sum([t_func(sk_chain[i].position) * (sk_chain[i+1].time-sk_chain[i].time) for i=1:length(sk_chain)-1])
#         temp = temp + (1/t_horizon) * (update_est - (sk_chain[end].time) * temp) 
#         estimates[i,j] = temp
#         x_init = sk_chain[end].position
#         print("Simulation progress: ", floor(j/n_batches*100), "% \r")
#     end
#   end
#   estimates
# end

function compare_samplers_parallel_imag(samplers::Vector{<:Function},
  n_batches::Int64,
  init_cond::Function)

  estimates = zeros(length(samplers),n_batches+1)
  final_post_mean = zeros(length(init_cond()),length(samplers))
  time_horizons = zeros(length(samplers),n_batches+1)
  println("")
  Threads.@threads for (i,sampler) in collect(enumerate(samplers))
    x_init = init_cond()
    estimates[i,1] = mse(f0_vec,x_init)
    running_mean = zeros(length(x_init))
    t_horizon = 0 
    for j = 2:(n_batches+1)
        sk_chain = sampler(x_init)
        t_horizon += sk_chain[end].time
        time_horizons[i,j] = sk_chain[end].time
        update_est = sum([identity(sk_chain[k].position) * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
        running_mean = running_mean + (1/t_horizon) * (update_est - (sk_chain[end].time) * running_mean) 
        estimates[i,j] = mse(f0_vec,running_mean)
        x_init = sk_chain[end].position
        print("Simulation progress of $i: ", floor((j-1)/n_batches*100), "% \r")
    end
    # println("made it to end of $i")
    final_post_mean[:,i] = running_mean
  end
  (estimates,final_post_mean,time_horizons)
end

function compare_samplers_parallel_imag(samplers::Vector{<:Function},
  n_exp::Int64,
  n_batches::Int64,
  init_cond::Function)

  estimates = zeros(length(samplers),n_exp,n_batches+1)
  println("")
  Threads.@threads for n = 1:n_exp
    for (i,sampler) in collect(enumerate(samplers))
      print("Simulation progress: $n-th thread starting with sampler $i \r")
      x_init = init_cond()
      estimates[i,n,1] = mse(f0_vec,x_init)
      running_mean = zeros(length(x_init))
      t_horizon = 0 
      for j = 2:(n_batches+1)
          sk_chain = sampler(x_init)
          t_horizon += sk_chain[end].time
          update_est = sum([identity(sk_chain[k].position) * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
          running_mean = running_mean + (1/t_horizon) * (update_est - (sk_chain[end].time) * running_mean) 
          estimates[i,n,j] = mse(f0_vec,running_mean)
          x_init = sk_chain[end].position
      end
    end
  end
  estimates
end

function compare_zzsamplers_parallel_imag(samplers::Vector{<:Function},
  n_exp::Int64,
  n_batches::Int64,
  init_cond::Function)

  mse_mean = zeros(length(samplers),n_exp,n_batches+1)
  mse_var = zeros(length(samplers),n_exp,n_batches+1)
  final_post_mean = zeros(length(f0),length(samplers))
  final_post_var = zeros(length(f0),length(samplers))
  println("")
  Threads.@threads for n = 1:n_exp
    for (i,sampler) in collect(enumerate(samplers))
      print("Simulation progress: $n-th thread starting with sampler $i \r")
      (x_init,v_init) = init_cond()
      mse_mean[i,n,1] = mse(f0_vec,x_init)
      running_mean = zeros(length(x_init))
      running_var_mat = zeros(length(x_init))
      mse_var[i,n,1] = mse(var_MYMALA,reshape(running_var_mat,size(var_MYMALA)))
      t_horizon = 0 
      for j = 2:(n_batches+1)
          sk_chain = sampler(x_init,v_init)
          t_horizon += sk_chain[end].time
          update_est = sum([sk_chain[k].position * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
          running_mean = running_mean + (1/t_horizon) * (update_est - (sk_chain[end].time) * running_mean) 
          # running_cov_mat = running_cov_mat + sum([reshape(sk_chain[k].position,size(var_MYMALA) )*transpose(reshape(sk_chain[k].position,size(var_MYMALA) )) * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
          running_var_mat = running_var_mat + sum([sk_chain[k].position.^2 * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
          mse_mean[i,n,j] = mse(f0_vec,running_mean)
          mse_var[i,n,j] = mse(vec(var_MYMALA),running_var_mat/t_horizon - running_mean.^2)
          (x_init,v_init) = (sk_chain[end].position,sk_chain[end].velocity)
      end
      if n == 1
        final_post_mean[:,i] = running_mean
        final_post_var[:,i] = running_var_mat/t_horizon - running_mean.^2
      end
    end
  end
  # (mse_mean,mse_var)
  (mse_mean,mse_var,final_post_mean,final_post_var)
end

function compare_zzsamplers_parallel_imag(samplers::Vector{<:Function},
  n_batches::Int64,
  init_cond::Function)

  mse_mean = zeros(length(samplers),n_batches+1)
  mse_var = zeros(length(samplers),n_batches+1)
  final_post_mean = zeros(length(f0),length(samplers))
  final_post_var = zeros(length(f0),length(samplers))
  time_horizons = zeros(length(samplers))
  println("")
  Threads.@threads for (i,sampler) in collect(enumerate(samplers))
    (x_init,v_init) = init_cond()
    mse_mean[i,1] = mse(f0_vec,x_init)
    running_mean = zeros(length(x_init))
    # running_var_mat = zeros(size(var_MYMALA))
    running_var_mat = zeros(length(x_init))
    mse_var[i,1] = mse(var_MYMALA,reshape(running_var_mat,size(var_MYMALA)))
    t_horizon = 0 
    for j = 2:(n_batches+1)
        sk_chain = sampler(x_init,v_init)
        t_horizon += sk_chain[end].time
        update_est = sum([sk_chain[k].position * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
        running_mean = running_mean + (1/t_horizon) * (update_est - (sk_chain[end].time) * running_mean) 
        # running_var_mat = running_var_mat + sum([reshape(sk_chain[k].position,size(var_MYMALA)).^2 * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
        running_var_mat = running_var_mat + sum([sk_chain[k].position.^2 * (sk_chain[k+1].time-sk_chain[k].time) for k=1:length(sk_chain)-1])
        mse_mean[i,j] = mse(f0_vec,running_mean)
        mse_var[i,j] = mse(vec(var_MYMALA),running_var_mat/t_horizon - running_mean.^2)
        # mse_var[i,j] = mse(var_MYMALA,running_var_mat/t_horizon - reshape(running_mean,size(var_MYMALA)).^2)
        (x_init,v_init) = (sk_chain[end].position,sk_chain[end].velocity)
        print("Simulation progress of $i: ", floor((j-1)/n_batches*100), "% \r")
    end
    println("made it to end of $i")
    time_horizons[i] = t_horizon
    final_post_mean[:,i] = running_mean
    final_post_var[:,i] = running_var_mat/t_horizon - running_mean.^2
  end
  (mse_mean,mse_var,final_post_mean,final_post_var,time_horizons)
end