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

# # Lab Frame Observables for an Adiabatic Interferometric Scheme 

# Here, we explore the expectation values of the full interferometer for a fully adiabatic time evolution.

using QuantumPropagators
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter
using FromFile

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
const SEPARATION_TIME = 0.1sec;
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;
const MOMENTUM_TARGET = - EFFECTIVE_MASS * OMEGA_TARGET;

includet("./include/rotating_tai.jl")

includet("./include/split_propagator.jl")

includet("./include/free_propagator.jl")

includet("./include/position_momentum_observables.jl")

includet("./include/propagate_scheme.jl")

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * cos(π * t / (2t_r))^2;

theta_grid = collect(range(0, 0.25π, length=1024));

args = Dict{Symbol,Any}(
    :theta_grid => theta_grid,
    :potential_depth => POTENTIAL_DEPTH,
    :omega_up => omega_ramp_up,
    :omega_down => omega_ramp_down,
    :omega_0 => OMEGA_TARGET,
    :t_r => SEPARATION_TIME,
    :n_cycles => 2,
    :nt_free => 10_000,
)

# ## Ideal dynamics (Ω=0)

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args...,
    parallel=true,
    ret=:expvals,
    frame,
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left, expvals_right;
    show_standard_deviation=true,
    frame
)

# +
frame=:moving

tlists, omega_vals, expvals_left_Ω0, expvals_right_Ω0 = propagate_scheme(;
    args...,
    parallel=true,
    ret=:expvals,
    frame
);

_, _, states_left_Ω0, states_right_Ω0 = propagate_scheme(;
    args...,
    parallel=true,
    ret=:states,
    frame
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left_Ω0, expvals_right_Ω0;
    show_standard_deviation=false,
    frame
)
# -

# the fast oscillations do not depend on the propagation method!

# ## Response to Ω ≠ 0

using QuantumControlBase: @threadsif

"""Evaluate the final "right" population depending on Ω."""
function scan_signal(; parallel=1, n_samples=21, n_cycles=2, Ω_max = (0.5 / n_cycles) / sec, kwargs...)
    if parallel ≡ true
        parallel=1
    end
    Ω_vals = collect(range(0, Ω_max; length=n_samples))
    P_vals = zeros(n_samples)
    @threadsif (parallel ≥ 1) for i=1:n_samples
        P_vals[i] = propagate_scheme(;
            Ω=Ω_vals[i],
            n_cycles,
            parallel=(parallel ≥ 2),
            ret=:P_right,
            kwargs...
        )
    end
    return Ω_vals, P_vals
end

Ω_vals, P_vals = scan_signal(;
    args...,
    parallel=2,
);

plot(Ω_vals / (π/sec), P_vals; label="", xlabel="Ω (π/sec)", ylabel="population (right)")

function contrast(populations)
    P_max = maximum(populations)
    P_min = minimum(populations)
    return (P_max - P_min) / (P_max + P_min)
end;

contrast(P_vals)

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args...,
    parallel=true,
    Ω=0.02π/sec,
    ret=:expvals,
    frame
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left, expvals_right;
    show_standard_deviation=true,
    frame
)

# +
frame=:moving

tlists, omega_vals, expvals_left_Ω1, expvals_right_Ω1 = propagate_scheme(;
    args...,
    parallel=true,
    Ω=0.02π/sec,
    ret=:expvals,
    frame
);

_, _, states_left_Ω1, states_right_Ω1 = propagate_scheme(;
    args...,
    parallel=true,
    Ω=0.02π/sec,
    ret=:states,
    frame
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left_Ω1, expvals_right_Ω1;
    show_standard_deviation=false,
    frame
)
# -

gr()

signal_pos_up = expvals_left_Ω1[1][1,:] .- 0.125π;

plot(signal_pos_up)

plot(signal_pos_up[1:1000]; marker=true)

plot(signal_pos_up[end-1000:end]; marker=true)

# ## Final state with and without Ω

P_right_Ω1 = propagate_scheme(;
    args...,
    parallel=true,
    Ω=0.02π/sec,
    ret=:P_right,
)

Ψ_left_Ω0 = states_left_Ω0[3][:,end];
Ψ_left_Ω1 = states_left_Ω1[3][:,end];
Ψ_right_Ω0 = states_right_Ω0[3][:,end];
Ψ_right_Ω1 = states_right_Ω1[3][:,end];

plot(theta_grid ./ π, abs2.(Ψ_right_Ω0), label="Ω=0")
plot!(theta_grid ./ π, abs2.(Ψ_right_Ω1), label="Ω≠0")
plot!(;xlim=(0.1, 0.15))

abs(Ψ_right_Ω1 ⋅ Ψ_right_Ω0)

Δϕ_right = angle(Ψ_right_Ω1 ⋅ Ψ_right_Ω0)
Δϕ_right / π

abs(Ψ_left_Ω1 ⋅ Ψ_left_Ω0)

Δϕ_left = angle(Ψ_left_Ω1 ⋅ Ψ_left_Ω0)
Δϕ_left / π

Δϕ = Δϕ_right - Δϕ_left
cos(Δϕ/2)^2

cos(Δϕ/2)^2 - P_right_Ω1
