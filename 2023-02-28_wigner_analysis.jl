# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---



# # Analyzing the dynamics via a Wigner plot

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
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const MOMENTUM_TARGET = -EFFECTIVE_MASS * OMEGA_TARGET;

projectdir(args...) = joinpath(@__DIR__, args...)
#plotsdir(args...) = projectdir("plots", "2023-02-07_continued_evolution", args...)
# rm(plotsdir(); recursive=true, force=true)
#mkpath(plotsdir())

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

# ## Propagation of Splitting

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

using QuantumPropagators.Controls: evaluate, discretize, discretize_on_midpoints

function propagate_splitting(;
    separation_time,
    evolution_time=0.0,
    potential_depth,
    omega_target=OMEGA_TARGET,
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
    T = separation_time + evolution_time
    nt = choose_timesteps(T; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, T, length=nt))


    function ω_func(t)
        if t <= separation_time
            omega_ramp_up(t; w0=omega_target, t_r=separation_time)
        else
            omega_target
        end
    end

    if ret == :omega
        return ω_func
    end

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

Ψ₀, θ = propagate_splitting(separation_time=1e-1sec, potential_depth=2.2MHz, ret=:initial_state);

Ψ = propagate_splitting(separation_time=1e-1sec, potential_depth=2.2MHz, ret=:propagation);

function psi_to_momentum(Ψ, θ)
    dθ = θ[2] - θ[1]
    nθ = length(θ)
    p::Vector{Float64} = fftshift(2π * fftfreq(nθ, 1 / dθ))
    Ψ_p = fftshift(fft(Ψ))
    return Ψ_p, p
end

# ## Implementation of Wigner transform

# +
"""Calculate Wigner transform W(p, x) from density matrix ρ(θ, θ′)."""
function wigner(ρ::Matrix)

    nx = size(ρ)[1]
    np = nx - 1
    @assert mod(nx, 2) == 0  # "Even grid" case
    X = copy(ρ)  # uneven grid would have to be padded in addition to copy
    W = zeros(ComplexF64, np, nx)
    vtmp = zeros(ComplexF64, np)
    h = nx ÷ 2
    hp = np ÷ 2
    FFT! = plan_fft!(vtmp)

    for ix = 1:nx

        v = @view W[:, ix]
        # Add counter-diagonal elements of the matrix. However, every second
        # counter-diagonal (the even ones) are omitted. This leads to a loss
        # of phasespace resolution but in general, this method is sufficient.
        if ix ≤ h  # upper left triangle
            for jp = (h-(ix-1)):(h+(ix-1))  # along the counter-diagonals
                v[jp] = X[ix+(h-jp), ix-(h-jp)]
            end
        else # lower right triangle
            for jp = (ix-h):nx-(ix-h)
                v[jp] = X[ix+(h-jp), ix-(h-jp)]
            end
        end

        # shuffle to avoid interferences in Fourier transform
        vtmp[:] .= v[:]
        v[1:hp+1] .= vtmp[hp+1:np]
        v[hp+2:np] .= vtmp[1:hp]

        # compute the forward complex discrete fourier transform of v
        FFT! * v
        fftshift!(vtmp, v)
        v[:] .= vtmp[:]

    end

    # Scale the Wigner matrix by 1/N due to DF
    lmul!(1 / np, W)

    return real.(W)

end

wigner(Ψ::Vector) = wigner(Ψ * Ψ')

# +
function plot_wigner(W::Matrix{Float64}, θ::Vector{Float64}; kwargs...)
    dθ = θ[2] - θ[1]
    nθ = length(θ)
    p::Vector{Float64} = fftshift(2π * fftfreq(nθ, 1 / dθ))
    p = p[1:end-1]
    fig = heatmap(
        θ ./ π, p, W;
        xlabel = "θ", ylabel = "p",
        kwargs...
    )
    return fig
end

plot_wigner(Ψ::Vector{ComplexF64}, args...; kwargs...) = plot_wigner(wigner(Ψ), args...; kwargs...)
# -

norm(Ψ₀)

# ## Wigner Transform of the Harmonic Oscillator Groundstate (Test)

"""Analytical ground state of harmonic potential"""
function psi_ho_groundstate(θ; θ₀=0.125π, V₀=2.2MHz, M=EFFECTIVE_MASS, m=N_SITES)
    ω = sqrt(V₀ * m^2 / M)
    a = 1 / sqrt(M * ω)
    c = 1 / (π^(1/4) * sqrt(a))
    Δθ = θ .- θ₀
    Ψ = c * exp.(-Δθ.^2 ./ (2 * a^2))
    return convert(Vector{ComplexF64}, Ψ)
end

"""Analytical Wigner function of harmonic potential groud state"""
function wigner_ho_groundstate(θ; θ₀=0.125π, V₀=2.2MHz, M=EFFECTIVE_MASS, m=N_SITES)
    dθ = θ[2] - θ[1]
    nθ = length(θ)
    p::Vector{Float64} = fftshift(2π * fftfreq(nθ, 1 / dθ))
    p = p[1:end-1]
    np = length(p)
    
    ω = sqrt(V₀ * m^2 / M)
    a = 1 / sqrt(M * ω)
    
    W = zeros(np, nθ)
    for iθ = 1:nθ
        for ip=1:np
            W[ip,iθ] = 4π * exp(-a^2 * p[ip]^2 - (θ[iθ]-θ₀)^2/a^2)
        end
    end
    return W
end

Ψ_ho_ground = psi_ho_groundstate(θ);

W_ho_ground = wigner_ho_groundstate(θ);

# +
function get_alpha(;V₀=2.2MHz, M=EFFECTIVE_MASS, m=N_SITES)
    α = 1 / (sqrt(M * V₀) * m * π)
end

plot_wigner(W_ho_ground, θ; ratio=get_alpha(), xlim=(0.12, 0.13), ylim=(-500, 500))
# -

(Ψ_ho_ground' * Ψ_ho_ground) * (θ[2] - θ[1])

W_Ψ_ho_ground = wigner(Ψ_ho_ground)
plot_wigner(W_Ψ_ho_ground, θ;  α=1, ratio=get_alpha()/2, xlim=(0.12, 0.13), ylim=(-1500, 1500))

# ## Wigner transform of initial and final state

plot_wigner(Ψ₀, θ; ratio=get_alpha(V₀=2.2MHz)/2, xlim=(0.12, 0.13), ylim=(-1500, 1500))

plot_wigner(Ψ, θ; ratio=get_alpha(V₀=2.2MHz)/2, xlim=(0.12, 0.13), ylim=(-1500, 1500))

# Note: semi-classical phase-space dynamics of splitting plus free time evolution looks like this:
#
# ![](https://cdn.discordapp.com/attachments/827575316079443978/1080913439339319437/image.png)
