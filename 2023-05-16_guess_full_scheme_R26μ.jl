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

# #  Full dynamics for guess pulse at `t_r=150μs`, `V0=0.2MHz`, `R=25.46μm`

using QuantumPropagators
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter
using FromFile
using Printf

using Revise

using Plots: plot, plot!

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
const SEPARATION_TIME = 150μs;
const LOOP_TIME = 900ms;
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 0.2MHz;
const MOMENTUM_TARGET = - EFFECTIVE_MASS * OMEGA_TARGET;
const MOMENTUM_UNIT = EFFECTIVE_MASS * π / sec;


datadir(folders...) = joinpath(".", "data", "2023-05-16_guess_full_scheme", folders...)
mkpath(datadir())

includet("./include/rotating_tai.jl")

includet("./include/split_propagator.jl")

includet("./include/free_propagator.jl")

includet("./include/position_momentum_observables.jl")

includet("./include/propagate_scheme.jl")

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * cos(π * t / (2t_r))^2;

# ## Full Scheme Dynamics

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
    :initialize_with_Ω => true,
)

tlists, omega_vals, states_left, states_right = propagate_scheme(;
    args...,
    ret=:states,
    frame=:moving,
    parallel=true
);

plot(theta_grid./π, abs2.(states_left[1][:,1]))

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args...,
    ret=:expvals,
    parallel=true,
    frame,
);

df = collect_dynamics_dataframe(tlists, omega_vals, expvals_left; steps_up=10, steps_free=1, steps_down=10)

open(datadir("dynamics_guess_lab.csv"), "w") do file
    println(file, join(names(df), ","))
    df_filtered = df[(df[!, "time (ms)"] .< 2) .| (df[!, "time (ms)"] .> 198), :]
    for row in eachrow(df_filtered)
        print(file, @sprintf("%.3f,", row[1]))
        println(file, join(map(v -> @sprintf("%.3e", v), row[2:end]), ","))
    end
end

plot_full_pos_mom_dynamics(
    tlists, expvals_left, expvals_right;
    show_standard_deviation=true,
    frame
)


# +
frame=:moving

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args...,
    ret=:expvals,
    parallel=true,
    frame,
);

df = collect_dynamics_dataframe(tlists, omega_vals, expvals_left; steps_up=10, steps_free=1, steps_down=10)

open(datadir("dynamics_guess_moving.csv"), "w") do file
    println(file, join(names(df), ","))
    df_filtered = df[(df[!, "time (ms)"] .< 2) .| (df[!, "time (ms)"] .> 198), :]
    for row in eachrow(df_filtered)
         print(file, @sprintf("%.3f,", row[1]))
        println(file, join(map(v -> @sprintf("%.3e", v), row[2:end]), ","))
    end
end

plot_full_pos_mom_dynamics(
    tlists, expvals_left, expvals_right;
    show_standard_deviation=true,
    frame
)
# -

# ## Response to Ω ≠ 0

using QuantumControlBase: @threadsif

"""Evaluate the final "right" population depending on Ω."""
function scan_signal(; parallel=1, n_samples=21, n_cycles=2, Ω_max = 4 * (0.5 / n_cycles) / sec, kwargs...)
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

open(datadir("sagnac_guess.csv"), "w") do file
    println(file, "Ω (π/sec),P_right")
    for (Ω_val, P_val) in zip(Ω_vals, P_vals)
        print(file, @sprintf("%.3e,", Ω_val / (π/sec)))
        println(file, @sprintf("%.3f", P_val))
    end
end

plot(Ω_vals / (π/sec), P_vals; label="", xlabel="Ω (π/sec)", ylabel="population (right)")

function contrast(populations)
    P_max = maximum(populations)
    P_min = minimum(populations)
    return (P_max - P_min) / (P_max + P_min)
end

contrast(P_vals)


