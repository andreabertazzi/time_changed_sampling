include("helper_tt.jl")
include("moves_samplers.jl")

function ULA_sim(∇U::Function,
    δ::Float64,
    N::Integer,
    x_init::Vector{Float64},
    )
    x = copy(x_init)
    states = Vector{Vector{Float64}}(undef,N+1)
    states[1] = x
    for n = 1:N
        x = x - ∇U(x)*δ + sqrt(2*δ)*randn(size(x));
        states[n+1] = x
    end
    states
end

function splitting_ABA(part_A!::Function,
                        part_B!::Function,
                        ∇U::Function,
                        δ::Float64,
                        N::Integer,
                        x_init::Vector{Float64},
                        v_init::AbstractVector)

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = x_init
    v = v_init
    for n = 1:N
        part_A!(∇U,x,v,δ/2)
        part_B!(∇U,x,v,δ)
        part_A!(∇U,x,v,δ/2)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    chain
end

function zzs_metropolised(target::Function,
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::Vector{Int64};
                         want_rej::Bool = false,
                         want_plot::Bool = false
                         )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) 
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        num = target(x) * exp(-δ*sum(switch_rate_new))
        den = target(x_old) * exp(-δ*sum(switch_rate_old))
        Z = rand(1)[1]
        if Z > num/den
            # println("Rejection!")
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    # println("")
    # println("Number of rejections: $n_rej")
    if want_plot
        iter = Integer(round(200/δ))
        # times = [chain[i].time for i = 1:iter]
        # positions = [chain[i].position[1] for i = 1:iter]
        # display(plot(times,positions))
        positions1 = [chain[i].position[1] for i = 1:iter]
        positions2 = [chain[i].position[2] for i = 1:iter]
        display(plot(positions1,positions2))
    end

    # chain
    if want_rej
        (chain,n_rej)
    else
        chain
    end

end

function zzs_metropolis_randstepsize(target::Function,
    ∇U::Function,
    avgstepsize::Float64,
    N::Integer,
    x_init::Vector{Float64},
    v_init::Vector{Int64}
    )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        v_old = copy(v)
        δ = draw_exponential_time(1/avgstepsize)
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) 
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        num = target(x) * exp(-δ*sum(switch_rate_new))
        den = target(x_old) * exp(-δ*sum(switch_rate_old))
        Z = rand(1)[1]
        if Z > num/den
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        push!(chain, skeleton(copy(x), copy(v), n * avgstepsize))
    end

    chain

end


function bps_metropolised(
                         target::Function,
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::AbstractVector;
                         unit_sphere::Bool = false,
                         want_rej::Bool = false,
                         want_plot::Bool = false
                         )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        scal_prod = dot(v,grad_x)
        switch_rate_old = max(0,scal_prod)
        reflect_given_rate!(v, switch_rate_old, grad_x, δ)
        switch_rate_new = max(0,-dot(v,grad_x))
        x = x + v * δ/2
        num = target(x) * exp(-δ*switch_rate_new)
        den = target(x_old) * exp(-δ*switch_rate_old)
        Z = rand(1)[1]
        if Z > num/den
            # println("Rejection!")
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    if want_plot
        # iter = Integer(round(200/δ))
        iter = 1000
        times = [chain[i].time for i = 1:iter]
        positions = [chain[i].position[2] for i = 1:iter]
        display(plot(times,positions))
        positions1 = [chain[i].position[1] for i = 1:iter]
        positions2 = [chain[i].position[2] for i = 1:iter]
        display(plot(positions1,positions2, label = "Metropolised BPS"))
    end

    if want_rej
        (chain,n_rej)
    else
        chain
    end

end

function splitting_zzs_DBD(
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::Vector{<:Real})

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) #define this function
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end

    chain

end


function splitting_bps_RDBDR_gauss(
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::AbstractVector;
                         unit_sphere::Bool = false,
                         want_plot::Bool = false)

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        x_old = copy(x)
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        scal_prod = dot(v,grad_x)
        switch_rate_old = max(0,scal_prod)
        reflect_given_rate!(v, switch_rate_old, grad_x, δ)
        switch_rate_new = max(0,-dot(v,grad_x))
        x = x + v * δ/2
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    if want_plot
        # iter = Integer(round(200/δ))
        iter = N
        # times = [chain[i].time for i = 1:iter]
        # positions = [chain[i].position[1] for i = 1:iter]
        # display(plot(times,positions))
        positions1 = [chain[i].position[1] for i = 1:iter]
        positions2 = [chain[i].position[2] for i = 1:iter]
        display(plot(positions1,positions2))
    end

    chain

end


function splitting_ABCBA(part_A!::Function,
                         part_B!::Function,
                         part_C!::Function,
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::AbstractVector)

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        part_A!(∇U,x,v,δ/2)
        part_B!(∇U,x,v,δ/2)
        part_C!(∇U,x,v,δ)
        part_B!(∇U,x,v,δ/2)
        part_A!(∇U,x,v,δ/2)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    chain

end

# function splitting_zzs_DBD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::AbstractVector)
#     splitting_ABA(flow_zzs!,jump_part_zzs!,∇U,δ,N,x_init,v_init)
# end

# Splittings of BPS

function splitting_bps_RDBDR(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(refreshment_part_bps!,flow_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_RBDBR(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(refreshment_part_bps!,reflection_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init)
end

# function splitting_bps_RDBDR_gauss(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
#     splitting_ABCBA(refreshment_part_bps_gauss!,flow_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init)
# end

function splitting_bps_DBRBD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(flow_bps!,reflection_part_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DBRBD_gauss(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(flow_bps!,reflection_part_bps!,refreshment_part_bps_gauss!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DRBRD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(flow_bps!,refreshment_part_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BRDRB(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(reflection_part_bps!,refreshment_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BDRDB(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(reflection_part_bps!,flow_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BDRDB_gauss(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(reflection_part_bps!,flow_bps!,refreshment_part_bps_gauss!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DBD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(flow_bps!,jump_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BDB(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(jump_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DR_B_DR(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(flow_and_refreshment_bps!,jump_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_B_DR_B(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(jump_part_bps!,flow_and_refreshment_bps!,∇U,δ,N,x_init,v_init)
end

function euler_pdmp_FD(flow::Function,
                    event_rates::Function,
                    jump::Function,
                    ∇U::Function,
                    δ::Float64,
                    N::Integer,
                    x_init::Vector{Float64},
                    v_init::AbstractVector)
    # Fully Discrete Euler approximation
    # Initial conditions required!
    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for i = 1 : N
        grad = ∇U(x)   # compute this as can re-use it. o.w. more than one grad computation per iteration :(
        rates = event_rates(grad,v) # can be scalar or vector
        t_event, i_event = event_times(rates)
        (x,v) = flow(x,v,δ)
        if t_event <= δ
            (x,v) = jump(grad,x,v,i_event)
        end
        push!(chain, skeleton(copy(x), copy(v), n*δ))
    end
    chain
end

function euler_pdmp_PD(flow::Function,
                    event_rates::Function,
                    jump::Function,
                    ∇U::Function,
                    δ::Float64,
                    N::Integer,
                    x_init::Vector{Float64},
                    v_init::AbstractVector)
    # Partially Discrete Euler approximation
    # Initial conditions required!
    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        grad = ∇U(x)
        rates = event_rates(grad,v) # can be scalar or vector
        t_event, i_event = event_times(rates)
        if t_event > δ
            (x,v) = flow(x,v,δ)
        else
            (x,v) = flow(x,v,t_event)
            (x,v) = jump(∇U,x,v,i_event)
            (x,v) = flow(x,v,δ-t_event)
        end
        push!(chain, skeleton(copy(x), copy(v), n*δ))
    end
    chain
end

function euler_zzs_FD(∇U::Function,δ::Float64,N::Integer,x_init::Vector{Float64},v_init::Vector{Real})
    euler_pdmp_FD(flow_zzs,event_rates_zzs,jump_zzs,∇U,δ,N,x_init,v_init)
end

function euler_zzs_PD(∇U::Function,δ::Float64,N::Integer,x_init::Vector{Float64},v_init::Vector{Int64})
    euler_pdmp_PD(flow_zzs,event_rates_zzs,jump_zzs,∇U,δ,N,x_init,v_init)
end

function euler_bps_PD(∇U::Function,δ::Float64,N::Integer,x_init::Vector{Float64},v_init::Vector{Float64})
    euler_pdmp_PD(flow_bps,event_rates_bps,jump_bps,∇U,δ,N,x_init,v_init)
end


tolerance = 1e-7

function BPS(∇E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0), refresh_rate::Float64 = 1.0)

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
    end

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;

    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    gradient = ∇E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    Δt_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)
        gradient = ∇E(x)

        if (!finished && Δt_switch_proposed < Δt_refresh)
            switch_rate = transpose(v) * gradient
            proposedSwitchIntensity = a
            if proposedSwitchIntensity < switch_rate - tolerance
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v = reflect(gradient,v)
                a = -switch_rate
                b = transpose(v) * Q * v
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update time to refresh
            Δt_refresh = Δt_refresh - Δt_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            a = transpose(v) * gradient
            b = transpose(v) * Q * v

            # update time to refresh
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            # push!(x_skeleton, x)
            # push!(v_skeleton, v)
            # push!(t_skeleton, t)
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end
        Δt_switch_proposed = switchingtime(a,b)
    end
    # println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    # println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)
    return skel_chain
end



function ZigZag(∇E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
    # ∂E(i,x) is the i-th partial derivative of the potential E, evaluated in x
    # Q is a symmetric matrix with nonnegative entries such that |(∇^2 E(x))_{ij}| <= Q_{ij} for all x, i, j
    # T is time horizon
    ∂E(i,x) = ∇E(x)[i]
    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = rand((-1,1), dim)
    end

    b = [norm(Q[:,i]) for i=1:dim];
    b = sqrt(dim)*b;

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    initial_gradient = [∂E(i,x) for i in 1:dim];
    a = v .* initial_gradient

    Δt_proposed_switches = switchingtime.(a,b)
    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        i = argmin(Δt_proposed_switches) # O(d)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed,Δt_excess);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_excess)
            switch_rate = v[i] * ∂E(i,x)
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = -switch_rate
                updateSkeleton = true
                accepted_switches += 1
            else
                a[i] = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_excess = Δt_excess - Δt_switch_proposed
            Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed
            Δt_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ∂E(i,x)

            # update upcoming event times
            Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
            Δt_excess = -log(rand())/(dim*excess_rate);
        end

        if updateSkeleton
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end

    end

    return skel_chain

end
