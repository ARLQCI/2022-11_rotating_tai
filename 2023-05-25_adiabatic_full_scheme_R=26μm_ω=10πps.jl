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

# # Lab Frame Observables for an Adiabatic Interferometric Scheme at ω=10π/ps

# This should reproduce Fig 4 in the paper

using QuantumPropagators
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter
using FromFile
using Printf
using QuantumControl: run_or_load

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
const OMEGA_TARGET = 10π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 2.2MHz;
const MOMENTUM_TARGET = -EFFECTIVE_MASS * OMEGA_TARGET;
const MOMENTUM_UNIT = EFFECTIVE_MASS * π / sec;

datadir(folders...) = joinpath(".", "data", "2023-05-25_adiabatic_ω=10πps", folders...)
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
    :nt_free => 10_000,
    :initialize_with_Ω => true,
)

# ## Benchmarking

# We compare the split propagator and Chebychev

# +
frame=:lab

@time propagate_scheme(;
    args...,
    n_cycles=1, # no free evolution
    parallel=true,
    ret=:expvals,
    method=:cheby,
    frame,
);
# -

@time propagate_scheme(;
    args...,
    n_cycles=1, # no free evolution
    parallel=true,
    ret=:expvals,
    method=:splitprop,
    frame,
);

# ## Ideal dynamics (Ω=0)

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args...,
    n_cycles=2,
    parallel=true,
    ret=:expvals,
    frame,
);
# -

map(length, tlists)

tlists[1][2] ./ μs

# +
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
    n_cycles=2,
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
    n_cycles=2,
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

# ## Response to Ω ≠ 0 (2 cycles)

using QuantumControlBase: @threadsif

"""Evaluate the final "right" population depending on Ω."""
function scan_signal(; parallel=1, n_samples=21, n_cycles, Ω_max, kwargs...)
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

Ω_vals_2cyc, P_vals_2cyc = run_or_load(datadir("sagnac_adiabatic_10πps_2cyc.jld2")) do
    scan_signal(;
        n_cycles=2,
        Ω_max=(0.667588/sec),
        n_samples=11,
        args...,
        parallel=2,
    )
end;

plot(Ω_vals_2cyc / (1/sec), 1 .- P_vals_2cyc; label="", marker=true, xlabel="Ω (rad/sec)", ylabel="population")

open(datadir("sagnac_adiabatic_10πps_2cyc.csv"), "w") do file
    println(file, "Ω (rad/s),|c₋|²")
    for (Ω_val, P_val) in zip(Ω_vals_2cyc, P_vals_2cyc)
        print(file, @sprintf("%.6f,", Ω_val / (1/sec)))
        println(file, @sprintf("%.6f", 1 - P_val))
    end
end

# ### Sagnac curve for paper

function sagnac_phase(Ω, ; Φ, R=TAI_RADIUS, M=RUBIDIUM_MASS)
    A = (R^2 / 2) * Φ
    return 4 * M * Ω * A
end

# position of peak:

function sagnac_peak(;R=TAI_RADIUS, M=RUBIDIUM_MASS, Φ=2π)
    """The value of Ω for which the Sagnac phase is π."""
    A = (R^2 / 2) * Φ
    return π / (4 * M * A)
end

function angular_displacement(omega_vals, tlists)
    dt(tlist) = tlist[2] - tlist[1]
    Φ = sum(omega_vals[1]) * dt(tlists[1])
    Φ += sum(omega_vals[2]) * dt(tlists[2])
    Φ += sum(omega_vals[3]) * dt(tlists[3])
    return Φ
end

function sagnac_population(
    Ω_vals;
    Φ=2π,
    R=TAI_RADIUS,
    M=RUBIDIUM_MASS,
)
    ΔΦ = sagnac_phase.(Ω_vals; Φ, R, M)
    return cos.(ΔΦ / 2) .^ 2
end

sagnac_peak() / (1/sec)

# Position of x-ticks in plot:

sagnac_peak() / (1e-2*π/sec)

2 * sagnac_peak() / (1e-2*π/sec)

open(datadir("sc_10πps_2cyc.csv"), "w") do file
    _Ω_vals = collect(range(0, 0.667588/sec, length=200))
    println(file, "Ω (rad/s),|c₋|²")
    for (Ω_val, P_val) in zip(_Ω_vals, sagnac_population(_Ω_vals))
        print(file, @sprintf("%.6f,", Ω_val / (1/sec)))
        println(file, @sprintf("%.6f", 1 - P_val))
    end
end

# ## Response to Ω ≠ 0 (9 cycles)

Ω_vals_9cyc, P_vals_9cyc = run_or_load(datadir("sagnac_adiabatic_10πps_9cyc.jld2")) do
    scan_signal(;
        n_cycles=9,
        Ω_max=(0.667588/sec),
        n_samples=86,
        args...,
        parallel=2,
    )
end;

plot(Ω_vals_9cyc / (1/sec), 1 .- P_vals_9cyc; label="", marker=true, xlabel="Ω (rad/sec)", ylabel="population")

open(datadir("sagnac_adiabatic_10πps_9cyc.csv"), "w") do file
    println(file, "Ω (rad/s),|c₋|²")
    for (Ω_val, P_val) in zip(Ω_vals_9cyc, P_vals_9cyc)
        print(file, @sprintf("%.6f,", Ω_val / (1/sec)))
        println(file, @sprintf("%.6f", 1 - P_val))
    end
end
