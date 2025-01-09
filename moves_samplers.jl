include("helper_split.jl")

## ZZS

function flow_zzs(x::Vector{Float64}, v::Vector{Int64}, s::Real)
    (x+v*s,v)
end

function flow_zzs(grad::Function, x::Vector{Float64}, v::Vector{Int64}, s::Real)
    flow_zzs(x,v,s)
end

function flow_zzs!(grad::Function, x::Vector{Float64}, v::Vector{Int64}, s::Real)
    x[:] = x + s*v
end

function jump_zzs(∇U::Function, x::Vector{Float64}, v::Vector{Int64}, i::Int64)
    return (x,jump_zzs(v, i))
end

function jump_zzs!(∇U::Function, x::Vector{Float64}, v::Vector{Int64}, i::Int64)
    v[i] = -v[i]
end

function jump_zzs(v::Vector{Int64}, i::Int64)
    flip!(v,i)
    v
end


function event_rates_zzs(grad::Vector{Float64}, v::Vector{Int64})
    # assuming refresh_rate is defined in the main environment!
    v.*grad
end

function jump_part_zzs(grad::Function, x::Vector{Float64}, v::Vector{Int64}, s::Real)
    rates = v .* grad(x)
    times = draw_exponential_time.(rates)
    for ℓ=1:length(x)
        if times[ℓ]<= s
            flip!(v,ℓ)
        end
    end
    return (x,v)
end

function jump_part_zzs!(grad::Function, x::Vector{Float64}, v::Vector{Int64}, s::Real)
    rates = v .* grad(x)
    times = draw_exponential_time.(rates)
    for ℓ=1:length(x)
        if times[ℓ]<= s
            flip!(v,ℓ)
        end
    end
end

function draw_velocity_zzs(dim::Int64)
    rand((-1, 1), dim)
end

function flip_given_rate!(v::AbstractVector, λ::Vector{Float64}, s::Real)
    event_time = draw_exponential_time.(λ)
    for i = 1:length(v)
        if event_time[i] <= s
            flip!(v, i)
            # v[i] = - copy(v[i])
        end
    end
end



## BPS

function flow_bps(x::Vector{Float64}, v::Vector{Float64}, s::Real)
    (x+v*s,v)
end

function flow_bps(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    flow_bps(x,v,s)
end

function jump_bps(grad::Vector{Float64}, v::Vector{Float64}, i::Int64)
    if i==1  # reflection
        return reflect(grad, v)
    else #refreshment
        return randn(length(v))
    end
end

function jump_bps(grad_fn::Function, x::Vector{Float64}, v::Vector{Float64}, i::Int64)
    if i==1  # reflection
        # grad = grad_fn(x)
        return (x,reflect(grad_fn(x), v))
    else #refreshment
        return (x,randn(length(v)))
    end
end

function event_rates_bps(grad::Vector{Float64}, v::Vector{Float64})
    # assuming refresh_rate is defined in the main environment!
    [dot(v,grad), refresh_rate]
end

function reflection_part_bps(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    grad_x = ∇U(x)
    reflection_time = draw_exponential_time(dot(grad_x,v))
    if reflection_time <= s
        v = reflect(grad_x,v)
    end
    (x,v)
end

function refreshment_part_bps(x::Vector{Float64}, v::Vector{Float64}, s::Real; unit_sphere::Bool = true)
    refresh_time = draw_exponential_time(refresh_rate)
    if refresh_time <= s
        v = draw_velocity_bps(length(v); unit_sphere = unit_sphere)
    end
    (x,v)
end

function refreshment_part_bps(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    refreshment_part_bps(x, v, s)
end

function jump_part_bps(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    grad_x = ∇U(x)
    reflection_time = draw_exponential_time(dot(grad_x,v))
    refresh_time = draw_exponential_time(refresh_rate)
    if min(reflection_time,refresh_time) <= s
        if reflection_time < refresh_time
            v = reflect(grad_x,v)
        else
            v = draw_velocity_bps(length(v))
        end
    end
    (x,v)
end

function draw_velocity_bps(d::Int64; unit_sphere::Bool = true)
    if unit_sphere
        a = randn(d)
        a/norm(a)
    else
        randn(d)
    end
end

function draw_velocity_gauss(d::Int64)
    randn(d)
end

function initial_state_bps(d::Int64)
    x = rand(d) - 0.5*ones(d)
    v = randn(d)
    # if velocity on unit sphere
    v = v/norm(v)
    (x,v)
end

function initial_state_bps_gaussianpos(d::Int64; unitsphere::Bool = true)
    x = randn(d)
    v = randn(d)
    if unitsphere
        v = v/norm(v)
    end
    (x,v)
end

## functions! for BPS

function flow_bps!(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    x[:] = x + v*s
end

function refresh_bps!(v::Vector{Float64}, d::Int64; unit_sphere::Bool = true)
    if unit_sphere
        vel = randn(d)
        v[:] = vel/norm(vel)
    else
        v[:] = randn(d)
    end
end

function refresh_bps_gauss!(v::Vector{Float64}, d::Int64)
    v[:] = randn(d)
end

function reflect!(gradient::Vector{Float64}, v::Vector{Float64})
    v[:] -= 2 * (dot(gradient, v) / dot(gradient,gradient)) * gradient
end

function reflection_part_bps!(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    grad_x = ∇U(x)
    reflection_time = draw_exponential_time(dot(grad_x,v))
    if reflection_time <= s
        reflect!(grad_x,v)
    end
end

function refreshment_part_bps!(x::Vector{Float64}, v::Vector{Float64}, s::Real; unit_sphere::Bool = true)
    refresh_time = draw_exponential_time(refresh_rate)
    if refresh_time <= s
        refresh_bps!(v,length(v);unit_sphere=unit_sphere)
    end
end

function refreshment_part_bps!(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    refreshment_part_bps!(x, v, s)
end

function refreshment_part_bps_gauss!(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    refresh_time = draw_exponential_time(refresh_rate)
    if refresh_time <= s
        refresh_bps_gauss!(v,length(v))
    end
end

function refreshment_part_bps_gauss!(v::Vector{Float64}, s::Real)
    if draw_exponential_time(refresh_rate) <= s
        refresh_bps_gauss!(v,length(v))
    end
end

function jump_part_bps!(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    grad_x = ∇U(x)
    t = 0
    while t < s
        reflection_time = draw_exponential_time(dot(grad_x,v))
        refresh_time = draw_exponential_time(refresh_rate)
        if min(reflection_time,refresh_time) <= s-t
            if reflection_time < refresh_time
                t += reflection_time
                reflect!(grad_x,v)
            else
                t += refresh_time
                refresh_bps!(v,length(v))
            end
        else
            t = s+1
        end
    end
end

function flow_and_refreshment_bps!(∇U::Function, x::Vector{Float64}, v::Vector{Float64}, s::Real)
    t = 0
    while t < s
        refresh_time = draw_exponential_time(refresh_rate)
        if refresh_time <= s-t
            flow_bps!(∇U,x,v,refresh_time)
            refresh_bps!(v,length(v))
            t += refresh_time
        else
            flow_bps!(∇U,x,v,s-t)
            t = s + 1
        end
    end
end

function reflect_given_rate!(v::Vector{Float64}, λ::Real, grad_x::Vector{Float64}, s::Real)
    reflection_time = draw_exponential_time(λ)
    if reflection_time <= s
        reflect!(grad_x,v)
    end
end

## Rate functions

function rate_bps(grad::Vector{<:Real}, v::Vector{<:Real})
    max(0,dot(grad,v))
end

## Miscellaneous

function event_times(rates::Vector{Float64})
    times = draw_exponential_time.(rates)
    i = argmin(times)
    return (times[i],i)
end

