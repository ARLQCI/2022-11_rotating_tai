# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # Robustness of the full optimized scheme for `t_r=150μs`, `V0=0.2MHz`, `R=26μm`, `ω₀=50π/s`

using QuantumPropagators
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter
using FromFile
using Printf

Threads.nthreads()

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
const SEPARATION_TIME = 150μs;
const OMEGA_TARGET = 50π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const POTENTIAL_DEPTH = 0.2MHz;
const MOMENTUM_TARGET = -EFFECTIVE_MASS * OMEGA_TARGET;
const MOMENTUM_UNIT = EFFECTIVE_MASS * π / sec;


datadir(folders...) = joinpath(".", "data", "2023-12-04_optimized_full_scheme_robustness", folders...)
mkpath(datadir())

includet("./include/rotating_tai.jl")

includet("./include/split_propagator.jl")

includet("./include/free_propagator.jl")

includet("./include/position_momentum_observables.jl")

includet("./include/propagate_scheme.jl")

# ## Loading the Optimized Controls

using QuantumControl.Shapes: flattop
using QuantumControl: load_optimization

using FileIO: load

omega_ramp_up = load("./data/2023-05-17_OCT_tr=150μs_V0=0.2MHz_R=26μm_ω=50πps_opt_amplitude.npz")
omega_ramp_up ./ (π/sec)

plot(omega_ramp_up ./ (π/sec))

omega_ramp_down = reverse(omega_ramp_up);

plot(omega_ramp_down ./ (π/sec))

# ## Full Scheme Dynamics

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

theta_grid = collect(range(0, 0.25π, length=1024));

tlists, omega_vals, states_left, states_right = propagate_scheme(;
    args...,
    ret=:states,
    frame=:moving,
    parallel=true,
    verbose_displacement=true,
);

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args...,
    ret=:expvals,
    parallel=true,
    frame,
);

df = collect_dynamics_dataframe(tlists, omega_vals, expvals_left; steps_up=10, steps_free=1, steps_down=10)

open(datadir("dynamics_opt_lab.csv"), "w") do file
    println(file, join(names(df), ","))
    df_filtered = df[(df[!, "time (ms)"] .< 2) .| (df[!, "time (ms)"] .> 198), :]
    for row in eachrow(df_filtered)
        print(file, @sprintf("%.3f,", row[1]))
        println(file, join(map(v -> @sprintf("%.3e", v), row[2:end]), ","))
    end
end

open(datadir("dynamics_opt_lab_theta.csv"), "w") do file
    println(file, join([names(df)[1], names(df)[3], names(df)[4]], ","))
    i_row = 0
    for row in eachrow(df)
        i_row += 1
        if i_row == 1 || i_row == nrow(df) || (i_row % 10) == 1
            print(file, @sprintf("%.3f,", row[1]))
            print(file, @sprintf("%.3f,", row[3]))
            println(file, @sprintf("%.3e", row[4]))
        end
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

open(datadir("dynamics_opt_moving.csv"), "w") do file
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

# ## Response to Ω ≠ 0, with disturbances

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

disturbed_args = copy(args)
disturbed_args[:potential_depth] = 1.1 * POTENTIAL_DEPTH

disturbed_args[:potential_depth] / MHz

Ω_vals, P_vals_disturbed = scan_signal(;
    disturbed_args...,
    parallel=2,
);


Ψ₀ = propagate_scheme(;
    ret=:initial_state,
    args...
);

Ω_vals, P_vals_disturbed_Ψ = scan_signal(;
    disturbed_args...,
    parallel=2,
    Ψ₀
);

function eigenstate_disturbance(V0_vals; Ψ₀=Ψ₀, args=args)
    result = Float64[]
    for V0 in V0_vals
        _disturbed_args = copy(args)
        _disturbed_args[:potential_depth] = V0
        _Ψ₀ = propagate_scheme(;
            ret=:initial_state,
            _disturbed_args...
        );
        push!(result, abs2(Ψ₀ ⋅ _Ψ₀))
    end
    return result
end

V0_vals = collect(range(0.2MHz, 2.2MHz, length=20));

Ψ_overlaps = eigenstate_disturbance(V0_vals);

plot(V0_vals ./ MHz, Ψ_overlaps; label="Overlap with eigenstate at V₀=0.2MHz", xlabel="V₀ (MHz)")

open(datadir("sagnac_opt_disturbed.csv"), "w") do file
    println(file, "Ω (π/sec),P_right,P_right_Ψ")
    for (Ω_val, P_val, P_val_Ψ) in zip(Ω_vals, P_vals_disturbed, P_vals_disturbed_Ψ)
        print(file, @sprintf("%.3e,", Ω_val / (π/sec)))
        println(file, @sprintf("%.3f,", P_val))
        println(file, @sprintf("%.3f", P_val_Ψ))
    end
end

using CSV
using DataFrames

guess_data = CSV.read("./data/2023-05-16_guess_full_scheme/sagnac_guess.csv", DataFrame);

opt_data = CSV.read("./data/2023-05-17_optimized_full_scheme/sagnac_opt.csv", DataFrame);



plot(Ω_vals / (π/sec), P_vals_disturbed; label="opt (10% V₀ dev)", xlabel="Ω (π/sec)", ylabel="population (right)", marker=true)
plot!(Ω_vals / (π/sec), P_vals_disturbed_Ψ; label="opt (10% V₀ dev; Ψ)")
plot!(opt_data[!, "Ω (π/sec)"], opt_data[!, "P_right"] , label="opt")
plot!(guess_data[!, "Ω (π/sec)"], guess_data[!, "P_right"] , label="guess")

function contrast(populations)
    P_max = maximum(populations)
    P_min = minimum(populations)
    return (P_max - P_min) / (P_max + P_min)
end;

contrast(opt_data[!, "P_right"])

contrast(P_vals_disturbed)

contrast(P_vals_disturbed_Ψ)

contrast(guess_data[!, "P_right"])
