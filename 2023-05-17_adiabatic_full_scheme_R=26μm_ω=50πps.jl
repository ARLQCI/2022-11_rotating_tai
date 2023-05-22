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

using QuantumPropagators
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter
using FromFile
using Printf

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
const TAI_RADIUS = 25.46μm
const N_SITES = 8;
const SEPARATION_TIME = 0.1sec;
const OMEGA_TARGET = 50π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;
const MOMENTUM_TARGET = -EFFECTIVE_MASS * OMEGA_TARGET;
const MOMENTUM_UNIT = EFFECTIVE_MASS * π / sec;

datadir(folders...) = joinpath(".", "data", "2023-05-17_adiabatic_full_scheme", folders...)

mkpath(datadir())

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
    :n_cycles => 10,
    :nt_free => 10_000,
    :initialize_with_Ω => true,
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


df = collect_dynamics_dataframe(tlists, omega_vals, expvals_left; steps_up=1000, steps_free=100, steps_down=1000)

open(datadir("dynamics_adiabatic_lab.csv"), "w") do file
    println(file, join(names(df), ","))
    for row in eachrow(df)
        println(file, join(map(v -> @sprintf("%.2e", v), row), ","))
    end
end

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

df = collect_dynamics_dataframe(tlists, omega_vals, expvals_left_Ω0; steps_up=1000, steps_free=100, steps_down=1000)

open(datadir("dynamics_adiabatic_moving.csv"), "w") do file
    println(file, join(names(df), ","))
    for row in eachrow(df)
        println(file, join(map(v -> @sprintf("%.2e", v), row), ","))
    end
end

_, _, states_left_Ω0, states_right_Ω0 = propagate_scheme(;
    args...,
    parallel=true,
    ret=:states,
    frame
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left_Ω0, expvals_right_Ω0;
    show_standard_deviation=true,
    frame
)
# -

# ## Response to Ω ≠ 0

using QuantumControlBase: @threadsif

"""Evaluate the final "right" population depending on Ω."""
function scan_signal(; parallel=1, n_samples=21, n_cycles=10, Ω_max = 4 * (0.5 / n_cycles) / sec, kwargs...)
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
