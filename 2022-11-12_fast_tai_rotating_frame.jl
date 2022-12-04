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
#     display_name: Julia 1.8 (4 threads)
#     language: julia
#     name: julia-1.8-multithread
# ---

# # Rotating TAI with ω(t) - fast splitting in the moving frame

# Here, we use $\omega(t)$ as the control field and $\phi(t) = \int \omega(t) dt$ as the control amplitude.
#
# We also use the Hamiltonian in the moving frame,
#
# \begin{equation}
# \tilde{H}_{\pm} (t)
# = -\frac{\hbar^2}{2mR^2}\frac{\partial^2}{\partial \theta^2} + V_0 \cos\left(m \theta\right) + i \hbar \omega_{\pm}(t) \frac{\partial}{\partial \theta}
# = \left((\hat{T} - \Omega \, \hat{p})  \mp \omega(t) \, \hat{p}\right) + \hat{V}
# \end{equation}

# ## Hamiltonian

using QuantumPropagators
using LinearAlgebra
using FFTW
using ProgressMeter

import QuantumControl.Controls: discretize, discretize_on_midpoints, evaluate

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
const SEPARATION_TIME = 10ms;
const LOOP_TIME = 900ms;
const OMEGA_TARGET = 200π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;

includet("./rotating_tai.jl")


includet("./split_propagator.jl")

tlist = collect(range(0, SEPARATION_TIME, length=(Int(SEPARATION_TIME ÷ ms) * 1000 + 1)));
theta_grid = collect(range(0, 0.25π, length=2048));

length(tlist)

# **TODO: choose number of time grid points so that cheby is stable and split-prop gives the same result**

function rotating_tai_hamiltonian(;
    tlist,
    θ,
    ω,
    Ω=0.0,
    direction=1,
    V₀=POTENTIAL_DEPTH,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)

    V = Diagonal(V₀ .* cos.(m .* θ))

    dθ = θ[2] - θ[1]
    nθ = length(θ)
    pgrid::Vector{Float64} = 2π * fftfreq(nθ, 1 / dθ)
    P = Diagonal(pgrid)
    K = Diagonal(pgrid .^ 2 / (2 * mass))

    _Ψ = Array{ComplexF64}(undef, nθ)
    fft_op = plan_fft!(_Ψ)
    ifft_op = plan_ifft!(_Ψ)
    transforms = (Ψ -> fft_op * Ψ, Ψ -> ifft_op * Ψ)

    K′ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = SplitGenerator(K′ + ω * P, V, transforms...)
        elseif direction == -1
            H = SplitGenerator(K′ - ω * P, V, transforms...)
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = SplitGenerator(hamiltonian(K′, (P, ω)), V, transforms...)
        elseif direction == -1
            H = SplitGenerator(hamiltonian(K′, (-P, ω)), V, transforms...)
        else
            error("direction must be ±1")
        end
    end
end

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * cos(π * t / (2t_r))^2;


plot(
    tlist ./ sec,
    discretize(omega_ramp_up.(tlist), tlist) / (2π / sec);
    legend=:topleft,
    label="ω(t)",
    xlabel="time (sec)",
    ylabel="angular velocity (2π/sec)"
)

Ĥ = rotating_tai_hamiltonian(
    tlist=tlist,
    θ=theta_grid,
    ω=discretize_on_midpoints(omega_ramp_up, tlist)
);

typeof(Ĥ)

typeof(Ĥ.T)

typeof(Ĥ.V)

Ĥ₀ = evaluate(Ĥ, tlist, 1)
V̂₀ = Ĥ₀.V;

# ## Coordinate space representation

function p_matrix(theta_grid::Vector{Float64})
    nx::Int64 = length(theta_grid)
    dx::Float64 = theta_grid[2] - theta_grid[1]
    p̂ = zeros(ComplexF64, nx, nx)
    @inbounds for i = 1:nx
        for j = 1:nx
            if i ≠ j
                p̂[j, i] = ((-1)^(j - i)) / sin(π * (j - i) / nx)
            end
        end
    end
    lmul!(-1im * π / (nx * dx), p̂)
    return Hermitian(p̂)
end

function rotating_tai_hamiltonian_coord(;
    tlist,
    θ,
    ω,
    Ω=0.0,
    direction=1,
    V₀=POTENTIAL_DEPTH,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)

    V = Diagonal(V₀ .* cos.(m .* θ))

    dθ = θ[2] - θ[1]
    nθ = length(θ)
    P = p_matrix(θ)
    K = P^2 / (2 * mass)

    K′ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = K′ + ω * P + V
        elseif direction == -1
            H = K′ - ω * P + V
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = hamiltonian(K′ + V, (P, ω))
        elseif direction == -1
            H = hamiltonian(K′ + V, (-P, ω))
        else
            error("direction must be ±1")
        end
    end
    return H
end

# ## Calculate initial state

function get_ground_state(; θ, ω=0.0, Ω=0.0)
    H₀ = rotating_tai_hamiltonian_coord(; tlist=[0.0], θ, ω, Ω)
    eigensys = eigen(H₀)
    Ψ₀ = eigensys.vectors[:, 1]
    return Ψ₀
end

Ψ₀ = get_ground_state(; θ=theta_grid, ω=0.0);

plot(theta_grid ./ π, V̂₀.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀.diag ./ MHz)
plot!(
    theta_grid ./ π,
    50 .* abs2.(Ψ₀) .+ _offset,
    label="|Ψ₀|²",
    xlim=(0, 0.3),
    legend=:right
)

println("⟨θ⟩_Ψ₀ = $(round(abs(dot(Ψ₀, Diagonal(theta_grid), Ψ₀) / π), digits=10))π")

# ## Target state

Ψ_tgt = get_ground_state(; θ=theta_grid, ω=OMEGA_TARGET);

norm(Ψ_tgt - Ψ₀)

norm(abs2.(Ψ_tgt) - abs2.(Ψ₀))

plot(theta_grid ./ π, real.(Ψ_tgt), marker=true, xlim=(0.115, 0.135))

plot(theta_grid ./ π, real.(Ψ₀) - real.(Ψ_tgt), marker=true,  xlim=(0.115, 0.135))

# ## Propagation (Split Propagator)

split_states = propagate(
    Ψ₀,
    Ĥ,
    tlist;
    method=:splitprop,
    specrange_method=:arnoldi,
    storage=true,
    showprogress=true
);

function plot_system(generator, states, theta_grid, tlist, n; psi_target=nothing, psi_scale=50, kwargs...)
    t = tlist[n]
    Ĥ = generator
    V = evaluate(Ĥ, tlist, min(n, length(tlist) - 1)).V.diag
    offset = minimum(V / MHz)
    Ψ = states[:, n]
    fig = plot(theta_grid ./ π, V / MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
    plot!(fig, theta_grid ./ π, psi_scale * abs2.(Ψ) .+ offset, label="|Ψ|²")
    if !isnothing(psi_target)
        plot!(fig, theta_grid ./ π, psi_scale * abs2.(psi_target) .+ offset, label="tgt")
    end
    plot!(;title="t=$(t/ms)ms", kwargs...)
end

plot(theta_grid ./ π, real.(Ψ_tgt), label="tgt")
plot!(theta_grid ./ π, real.(split_states[:,end]), label="Ψ")
plot!(; xlim=(0.12, 0.13), ylim=(-0.5, 0.5))

anim = @animate for n = 1:(length(tlist)÷100):length(tlist)
    plot_system(Ĥ, split_states, theta_grid, tlist, n; xlim=(0.12, 0.13), ylim=(-20.0, 20.0), psi_target=Ψ_tgt)
end
gif(anim, "anim.gif", fps=10)

abs2(split_states[:, end] ⋅ Ψ_tgt)

angle(split_states[:, end] ⋅ Ψ_tgt) / π

# ## Free time evolution

Ψ_free = split_states[:, end];

V̂_free = evaluate(Ĥ, tlist, length(tlist) - 1).V;

plot(theta_grid ./ π, V̂_free.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂_free.diag ./ MHz)
plot!(theta_grid ./ π, 50 .* abs2.(Ψ_free) .+ _offset, label="|Ψ|²")

function free_time_evolution(Ψ::Vector{ComplexF64}, H::AbstractMatrix, Δt::Float64)
    eigensys = eigen(H)
    U = eigensys.vectors
    λ = eigensys.values
    Ψ_out = U * (exp.(-1im .* λ .* Δt) .* (U' * Ψ))
    return Ψ_out
end

H_loop = rotating_tai_hamiltonian_coord(; tlist, θ=theta_grid, ω=OMEGA_TARGET);

Ψ_free_out = free_time_evolution(Ψ_free, H_loop, LOOP_TIME);

# ## Full scheme propagation

# +
"""Propagate Ψ₀ defined on a single surface.

Ψ₀ may or may not be normalized. In general, the population on the surface,
‖Ψ₀‖², is preserved.
"""
function prop_scheme(;
    Ψ₀,
    ω₀,
    θ,
    t_r,
    t_loop,
    dir=1,
    Ω=0.0,
    V0=POTENTIAL_DEPTH,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)

    specrange_method = :arnoldi
    tlist_ramp_up = collect(range(0, t_r, length=(Int(t_r ÷ ms) * 1000 + 1)))
    Ĥ_ramp_up = rotating_tai_hamiltonian(;
        tlist=tlist_ramp_up,
        θ,
        Ω,
        ω=discretize_on_midpoints(t -> omega_ramp_up(t; w0=ω₀, t_r=t_r), tlist_ramp_up),
        direction=dir,
    )
    #t_loop = ...

    Ĥ_loop = rotating_tai_hamiltonian_coord(;
        tlist=[t_r, t_r + t_loop],
        θ,
        Ω,
        ω=ω₀,
        direction=dir,
    )

    tlist_ramp_down =
        collect(range(t_r + t_loop, 2 * t_r + t_loop, length=length(tlist_ramp_up)))
    Ĥ_ramp_down = rotating_tai_hamiltonian(;
        tlist=tlist_ramp_down,
        θ,
        Ω,
        ω=discretize_on_midpoints(
            t -> omega_ramp_down(t - t_r - t_loop; w0=ω₀, t_r=t_r),
            tlist_ramp_down
        ),
        direction=dir,
    )

    Ψ = propagate(Ψ₀, Ĥ_ramp_up, tlist_ramp_up; method=:splitprop, specrange_method)
    Ψ = free_time_evolution(Ψ, Ĥ_loop, t_loop)
    Ψ = propagate(Ψ, Ĥ_ramp_down, tlist_ramp_down; method=:splitprop, specrange_method)
    return Ψ

end
# -

function angular_displacement(; ω₀, t_r, t_loop)

    milliseconds = Int(Base.div(t_r, ms))
    nt = milliseconds * 1000 + 1
    tlist_ramp_up = collect(range(0, t_r, length=nt))
    dt_up = tlist_ramp_up[2] - tlist_ramp_up[1]
    ω_up = discretize_on_midpoints(t -> omega_ramp_up(t; w0=ω₀, t_r=t_r), tlist_ramp_up)

    tlist_ramp_down =
        collect(range(t_r + t_loop, 2 * t_r + t_loop, length=nt))
    dt_down = tlist_ramp_down[2] - tlist_ramp_down[1]
    ω_down = discretize_on_midpoints(
        t -> omega_ramp_down(t - t_r - t_loop; w0=ω₀, t_r=t_r),
        tlist_ramp_down
    )

    Φ_up = sum(ω_up) * dt_up
    Φ_loop = t_loop * ω₀
    Φ_down = sum(ω_down) * dt_down
    Φ = Φ_up + Φ_loop + Φ_down

    Φtgt = round(Φ / π) * π
    t_loop_opt = (Φtgt - (Φ_up + Φ_down)) / ω₀
    if abs(t_loop - t_loop_opt) > 1e-8
        @warn("angular displacement not aligned. Change t_loop to $(t_loop_opt/ms)ms")
    end

    return Φ
end

angular_displacement(ω₀=OMEGA_TARGET,t_r=SEPARATION_TIME, t_loop=LOOP_TIME) / π ###

surface_pop(Ψ) = norm(Ψ)^2

function eval_scheme(; ω₀, θ, t_r, t_loop, Ω)

    U_πhalf = [
        1/√2  𝕚/√2
        𝕚/√2  1/√2
    ]

    Ψright = get_ground_state(; θ, Ω)
    Ψleft = zeros(ComplexF64, length(Ψright))

    Ψright, Ψleft = U_πhalf * [Ψright, Ψleft]

    prop_right = Threads.@spawn prop_scheme(; Ψ₀=Ψright, ω₀, θ, t_r, t_loop, dir=1, Ω=Ω)
    prop_left = Threads.@spawn prop_scheme(; Ψ₀=Ψleft, ω₀, θ, t_r, t_loop, dir=-1, Ω=Ω)
    Ψright = fetch(prop_right)
    Ψleft = fetch(prop_left)

    Ψleft, Ψright = U_πhalf * [Ψleft, Ψright]
    return surface_pop(Ψright)
end

eval_scheme(ω₀=OMEGA_TARGET, θ=theta_grid, t_r=SEPARATION_TIME, t_loop=LOOP_TIME, Ω=0.0)

angular_displacement(ω₀=OMEGA_TARGET,t_r=SEPARATION_TIME, t_loop=LOOP_TIME) / π

function scan_Ω(Ω_list)
    P_list = Float64[]
    @showprogress for Ω in Ω_list
        P = eval_scheme(;
            ω₀=OMEGA_TARGET,
            θ=theta_grid,
            t_r=SEPARATION_TIME,
            t_loop=LOOP_TIME,
            Ω
        )
        push!(P_list, P)
    end
    return P_list
end

Ω_list = collect(range(0, 0.005 / sec, length=41))
P_list = scan_Ω(Ω_list)

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", legend=false)

function sagnac_phase(Ω, ; Φ, R=TAI_RADIUS, m=RUBIDIUM_MASS)
    A = (R^2/2) * Φ
    return 4 * m * Ω * A
end

function sagnac_population(
        Ω_list; R=TAI_RADIUS, m=RUBIDIUM_MASS, ω₀=OMEGA_TARGET, t_r=SEPARATION_TIME, t_loop=LOOP_TIME
    )
    Φ = angular_displacement(;ω₀, t_r, t_loop)
    ΔΦ = sagnac_phase.(Ω_list; Φ, R, m)
    return sin.(ΔΦ / 2).^2
end

plot(Ω_list / (1 / sec), P_list, xlabel="Ω (1/sec)", label="simulation", legend=:outertop)
plot!(Ω_list / (1 / sec), sagnac_population(Ω_list), label="sagnac")
plot!(;size=(600,450))
