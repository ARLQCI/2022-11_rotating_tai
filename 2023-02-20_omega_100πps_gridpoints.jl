# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Julia 1.8 (auto threads)
#     language: julia
#     name: julia-1.8
# ---

# # Choose grid points for ω₀=100π/s

# We need to determine how many grid points in $\theta$ are required to accurately represent ω₀=100π/s

using Revise

using Plots

const 𝕚 = 1im;
const μm = 1;
const μs = 1;
const ns = 1e-3μs;
const cm = 1e4μm;
const met = 1e6μm;
const sec = 1e6μs;
const ms = 1e3μs;
const MHz = 2π;
const Dalton = 1.5746097504353806e+01;

const RUBIDIUM_MASS = 86.91Dalton;
const TAI_RADIUS = 42μm
const N_SITES = 8;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;

using QuantumPropagators.Controls: evaluate, discretize, discretize_on_midpoints

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

function propagate_splitting(;
    separation_time,
    potential_depth,
    omega_target,
    number_of_sites=N_SITES,
    mass=EFFECTIVE_MASS,
    ret=:fidelity,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    theta_max=0.25π,
    theta_steps=1024,
    scale_potential=nothing,
    kwargs...
)
    nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, separation_time, length=nt))
    ω_func(t) = omega_ramp_up(t; w0=omega_target, t_r=separation_time)
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    if !isnothing(scale_potential)
        scale_potential = discretize_on_midpoints(scale_potential, tlist)
    end
    Ĥ = rotating_tai_hamiltonian(;
        tlist,
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
        scale_potential,
        ω=discretize_on_midpoints(ω_func, tlist)
    )
    if ret == :system
        return Ĥ, tlist
    end
    Ĥ₀ = evaluate(Ĥ, tlist, 1)
    Ψ₀ = get_ground_state(Ĥ₀, θ, π / 8, d=0.05, steps=10_000)
    if ret == :initial_state
        return Ψ₀, θ
    end
    Ĥ_tgt = evaluate(Ĥ, tlist, nt - 1)
    if ret == :H_tgt
        return Ĥ_tgt
    end
    Ψ_tgt = get_ground_state(Ĥ_tgt, θ, π / 8, d=0.05, steps=10_000)
    if ret == :target
        return Ψ_tgt, θ
    end
    Ψ = propagate(Ψ₀, Ĥ, tlist; method=:splitprop, kwargs...)
    if ret == :propagation
        return Ψ
    end
    F = abs2(Ψ ⋅ Ψ_tgt)
    if ret == :fidelity
        return F
    else
        error("Invalid ret=$ret")
    end
end

propagate_splitting(
    omega_target=100π/sec,
    separation_time=1e-7sec,
    potential_depth=0.2MHz,
    theta_steps=1024,
)

propagate_splitting(
    omega_target=100π/sec,
    separation_time=1e-7sec,
    potential_depth=0.2MHz,
    theta_steps=2048,
)

propagate_splitting(
    omega_target=100π/sec,
    separation_time=1e-1sec,
    potential_depth=2.2MHz,
    theta_steps=1024,
)

propagate_splitting(
    omega_target=100π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=1024,
)

propagate_splitting(
    omega_target=100π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=2048,
)

# It actually seems like 1024 is enough

# For optimal control, we might want to allow for it go even higher, though

# **500π/sec**

propagate_splitting(
    omega_target=500π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=1024,
)

propagate_splitting(
    omega_target=500π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=2048,
)

propagate_splitting(
    omega_target=500π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=4096,
)

# **1000π/sec**

propagate_splitting(
    omega_target=1000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=1024,
)

propagate_splitting(
    omega_target=1000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=2048,
)

propagate_splitting(
    omega_target=1000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=4096,
)

# **5000π/sec**

propagate_splitting(
    omega_target=5000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=1024,
)

propagate_splitting(
    omega_target=5000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=2048,
)

propagate_splitting(
    omega_target=5000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=4096,
)

propagate_splitting(
    omega_target=5000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=8192,
)

propagate_splitting(
    omega_target=5000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=16384,
)

propagate_splitting(
    omega_target=5000π/sec,
    separation_time=1e-3sec,
    potential_depth=2.2MHz,
    theta_steps=32768,
)

# For optimal control, we should use 2048 points, and not let ω go over 1000π/sec
