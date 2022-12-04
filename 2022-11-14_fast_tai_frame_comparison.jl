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

# # Frame Comparison with Fast Splitting

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
const SEPARATION_TIME = 1ms;
const LOOP_TIME = 900ms;
const OMEGA_TARGET = 1000π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;

includet("./integrated_amplitude.jl")

includet("./rotating_tai.jl")


includet("./split_propagator.jl")

tlist = collect(range(0, SEPARATION_TIME, length=(Int(SEPARATION_TIME ÷ ms) * 1000 + 1)));

theta_grid_lab_frame = collect(range(0, 0.75π, length=6144));

theta_grid_moving_frame = theta_grid_lab_frame[[θ <= 0.25π for θ ∈ theta_grid_lab_frame]];

length(tlist)

function rotating_tai_hamiltonian_lab_frame(;
    tlist,
    θ,
    ω::Vector{Float64},
    Ω=0.0,
    direction=1,
    V₀=POTENTIAL_DEPTH,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)
    @assert length(tlist) == length(ω) + 1
    @assert Ω == 0.0
    @assert direction == 1
    phi = IntegratedAmplitude(ω)
    V = RotTAI_PotentialGenerator(; V0=V₀, m=m, theta=θ, phi)
    dθ = θ[2] - θ[1]
    n_theta = length(θ)
    pgrid = 2π * fftfreq(n_theta, 1 / dθ)
    K = Diagonal(pgrid .^ 2 / (2 * mass))
    Ψrand = Array{ComplexF64}(undef, n_theta)
    fft_op = plan_fft!(Ψrand)
    ifft_op = plan_ifft!(Ψrand)
    SplitGenerator(K, V, Ψ -> fft_op * Ψ, Ψ -> ifft_op * Ψ)
end

function rotating_tai_hamiltonian_moving_frame(;
    tlist,
    θ,
    ω::Vector{Float64},
    Ω=0.0,
    direction=1,
    V₀=POTENTIAL_DEPTH,
    m=N_SITES,
    mass=EFFECTIVE_MASS
)
    @assert length(tlist) == length(ω) + 1
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

    K̃ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = SplitGenerator(K̃ + ω * P, V, transforms...)
        elseif direction == -1
            H = SplitGenerator(K̃ - ω * P, V, transforms...)
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = SplitGenerator(hamiltonian(K̃, (P, ω)), V, transforms...)
        elseif direction == -1
            H = SplitGenerator(hamiltonian(K̃, (-P, ω)), V, transforms...)
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

Ĥ_moving = rotating_tai_hamiltonian_moving_frame(
    tlist=tlist,
    θ=theta_grid_moving_frame,
    ω=discretize_on_midpoints(omega_ramp_up, tlist)
);

Ĥ_lab = rotating_tai_hamiltonian_lab_frame(
    tlist=tlist,
    θ=theta_grid_lab_frame,
    ω=discretize_on_midpoints(omega_ramp_up, tlist)
);

Ĥ₀_moving = evaluate(Ĥ_moving, tlist, 1)
V̂₀_moving = Ĥ₀_moving.V;

Ĥ₀_lab = evaluate(Ĥ_lab, tlist, 1)
V̂₀_lab = Ĥ₀_lab.V;

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

function rotating_tai_hamiltonian_coord_moving_frame(;
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

    K̃ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = K̃ + ω * P + V
        elseif direction == -1
            H = K̃ - ω * P + V
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = hamiltonian(K̃ + V, (P, ω))
        elseif direction == -1
            H = hamiltonian(K̃ + V, (-P, ω))
        else
            error("direction must be ±1")
        end
    end
    return H
end

# ## Calculate initial state

function get_ground_state_moving_frame(; θ, ω=0.0, Ω=0.0)
    H₀ = rotating_tai_hamiltonian_coord_moving_frame(; tlist=[0.0], θ, ω, Ω)
    eigensys = eigen(H₀)
    Ψ₀ = eigensys.vectors[:, 1]
    return Ψ₀
end

Ψ₀_moving = get_ground_state_moving_frame(; θ=theta_grid_moving_frame, ω=0.0);

plot(theta_grid_moving_frame ./ π, V̂₀_moving.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀_moving.diag ./ MHz)
plot!(
    theta_grid_moving_frame ./ π,
    50 .* abs2.(Ψ₀_moving) .+ _offset,
    label="|Ψ₀|²",
    xlim=(0, 0.3),
    legend=:right
)

Ψ₀_lab = get_ground_state(Ĥ₀_lab, theta_grid_lab_frame, 2π/16,  d=0.05, steps=10_000);

plot(theta_grid_lab_frame ./ π, V̂₀_lab.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀_lab.diag ./ MHz)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(Ψ₀_lab) .+ _offset,
    label="|Ψ₀|²",
    xlim=(0, 0.3),
    legend=:right
)

# ## Target state

Ψ_tgt_moving = get_ground_state_moving_frame(; θ=theta_grid_moving_frame, ω=OMEGA_TARGET);

Boost = Diagonal(exp.(1im .* EFFECTIVE_MASS .* OMEGA_TARGET .* theta_grid_lab_frame));

Ψ_tgt_lab = Boost * get_ground_state(
    evaluate(Ĥ_lab, tlist, length(tlist)-1),
    theta_grid_lab_frame, 0.6π,  d=0.05, steps=10_000);

# ## Propagation in lab frame

states_lab = propagate(
    Ψ₀_lab,
    Ĥ_lab,
    tlist;
    method=:splitprop,
    specrange_method=:arnoldi,
    storage=true,
    showprogress=true
);

plot(theta_grid_lab_frame ./ π, V̂₀_lab.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀_lab.diag ./ MHz)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(states_lab[:,end]) .+ _offset,
    label="|Ψ|²",
    xlim=(0.62, 0.63),
    legend=:right
)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(Ψ_tgt_lab) .+ _offset,
    label="tgt",
    legend=:right
)

abs2(states_lab[:,end] ⋅ Ψ_tgt_lab)

# ## Propagation in moving frame

states_moving = propagate(
    Ψ₀_moving,
    Ĥ_moving,
    tlist;
    method=:splitprop,
    specrange_method=:arnoldi,
    storage=true,
    showprogress=true
);

plot(theta_grid_moving_frame ./ π, V̂₀_moving.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀_moving.diag ./ MHz)
plot!(
    theta_grid_moving_frame ./ π,
    50 .* abs2.(states_moving[:,end]) .+ _offset,
    label="|Ψ|²",
    xlim=(0.12, 0.13),
)
plot!(
    theta_grid_moving_frame ./ π,
    50 .* abs2.(Ψ_tgt_moving) .+ _offset,
    label="tgt",
    legend=:top
)

abs2(states_moving[:,end] ⋅ Ψ_tgt_moving)

# ## Frame transformation

# +
function moving_to_lab(
    Ψ::Vector{ComplexF64},
    ω::Vector{Float64},
    θ_lab::Vector{Float64},
    tlist::Vector{Float64},
    n=length(tlist)
)
    dθ = θ_lab[2] - θ_lab[1]
    nθ = length(θ_lab)
    pgrid = 2π * fftfreq(nθ, 1 / dθ)
    P = Diagonal(pgrid)

    dt = tlist[2] - tlist[1]
    Φ = sum(ω[1:n-1]) * dt

    U = exp(-1im * Φ * P)

    Ψ = [Ψ..., zeros(length(θ_lab) - length(Ψ))...]
    fft!(Ψ)
    Ψ = U * Ψ
    ifft!(Ψ)
    return Ψ

end
# -

Ψ_lab = moving_to_lab(
    states_moving[:,end],
    get_controls(Ĥ_moving)[1],
    theta_grid_lab_frame,
    tlist
)
plot(theta_grid_lab_frame ./ π, V̂₀_lab.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀_lab.diag ./ MHz)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(Ψ_lab) .+ _offset,
    label="|Ψ|²",
    #xlim=(0.62, 0.63),
    legend=:top
)

plot(theta_grid_lab_frame ./ π, V̂₀_lab.diag ./ MHz, xlabel="θ/π", ylabel="Energy (MHz)", label="V")
_offset = minimum(V̂₀_lab.diag ./ MHz)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(states_lab[:,end]) .+ _offset,
    label="|Ψ|²",
    xlim=(0.62, 0.63),
    legend=:top
)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(Ψ_lab) .+ _offset,
    label="|Ψ(from moving)|²",
)
plot!(
    theta_grid_lab_frame ./ π,
    50 .* abs2.(Ψ_tgt_lab) .+ _offset,
    label="tgt",
)
