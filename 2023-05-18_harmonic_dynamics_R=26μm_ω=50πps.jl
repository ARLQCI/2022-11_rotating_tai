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

# # Harmonic approximation

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
const OMEGA_TARGET = 50π / sec;
const EFFECTIVE_MASS = TAI_RADIUS^2 * RUBIDIUM_MASS;
const MOMENTUM_TARGET = - EFFECTIVE_MASS * OMEGA_TARGET;
const MOMENTUM_UNIT = EFFECTIVE_MASS * π / sec;

using Revise
using Plots: plot, plot!, vline!

includet("./include/rotating_tai.jl")
includet("./include/split_propagator.jl")
includet("./include/free_propagator.jl")
includet("./include/position_momentum_observables.jl")
includet("./include/propagate_scheme.jl")

# ## Ground state (nonadiabatic)

theta_grid = collect(range(0, 0.25π, length=1024));

Ψ0 = propagate_scheme(;
    theta_grid,
    ret=:initial_state,
    direction=1,
    omega_up=t->0.0,
    omega_down=t->0.0,
    omega_0=0.0,
    t_r=150μs,
    n_cycles=1,
    potential_depth=0.2MHz,
    mass=EFFECTIVE_MASS,
    number_of_sites=N_SITES,
);

plot(abs2.(Ψ0))

"""Analytical ground state of harmonic potential"""
function psi_ho_groundstate(θ; θ₀=0.125π, V₀=2.2MHz, M=EFFECTIVE_MASS, m=N_SITES)
    ω = sqrt(V₀ * m^2 / M)
    a = 1 / sqrt(M * ω)
    c = 1 / (π^(1/4) * sqrt(a))
    Δθ = θ .- θ₀
    Ψ = c * exp.(-Δθ.^2 ./ (2 * a^2))
    return convert(Vector{ComplexF64}, Ψ)
end

Ψ_ho = psi_ho_groundstate(theta_grid);

# Note that the analytic ground state is normalized differently than the numberic ground state (integral normalization vs vector normalization)

plot(theta_grid ./ π, abs2.(Ψ_ho ./ norm(Ψ_ho));
    label="Ψ (analytical)",
    xlabel="θ (π)",
    ylabel="amplitude",
    xlim=(0.1, 0.14)
)
plot!(theta_grid ./ π, abs2.(Ψ0);
    label="Ψ (numerical)",
)

plot(theta_grid ./ π, abs2.(Ψ_ho ./ norm(Ψ_ho)) .- abs2.(Ψ0 ./ norm(Ψ0));
    label="ΔΨ",
    xlabel="θ (π)",
    ylabel="amplitude",
    xlim=(0.1, 0.14)
)

1 - real((Ψ_ho ./ norm(Ψ_ho)) ⋅ Ψ0)

# ## Harmonic adiabatic evolution

# +
omega_ramp_up_adiabatic(t; w0=OMEGA_TARGET, t_r=0.1sec) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down_adiabatic(t; w0=OMEGA_TARGET, t_r=0.1sec) = w0 * cos(π * t / (2t_r))^2;

args_adiabatic = Dict{Symbol,Any}(
    :theta_grid => theta_grid,
    :potential_depth => 2.2MHz,
    :omega_up => omega_ramp_up_adiabatic,
    :omega_down => omega_ramp_down_adiabatic,
    :omega_0 => OMEGA_TARGET,
    :t_r => 0.1sec,
    :n_cycles => 10,
    :nt_free => 10000,
    :model => :harmonic,
)

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args_adiabatic...,
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

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args_adiabatic...,
    parallel=true,
    ret=:expvals,
    frame,
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left, expvals_right;
    show_standard_deviation=false,
    frame
)

# +
frame=:mixed

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args_adiabatic...,
    parallel=true,
    ret=:expvals,
    frame,
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left, expvals_right;
    show_standard_deviation=false,
    frame
)
# -

# ### Response to Ω ≠ 0

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
    args_adiabatic...,
    parallel=2,
);

function sagnac_phase(Ω, ; Φ, R=TAI_RADIUS, M=RUBIDIUM_MASS)
    A = (R^2 / 2) * Φ
    return 4 * M * Ω * A
end

function angular_displacement(omega_vals, tlists)
    dt(tlist) = tlist[2] - tlist[1]
    Φ = sum(omega_vals[1]) * dt(tlists[1])
    Φ += sum(omega_vals[2]) * dt(tlists[2])
    Φ += sum(omega_vals[3]) * dt(tlists[3])
    return Φ
end

Φ = angular_displacement(omega_vals, tlists);
Φ ./ π

function sagnac_population(
    Φ,
    Ω_vals;
    R=TAI_RADIUS,
    M=RUBIDIUM_MASS,
)
    ΔΦ = sagnac_phase.(Ω_vals; Φ, R, M)
    return cos.(ΔΦ / 2) .^ 2
end

plot(Ω_vals / (π/sec), P_vals; label="Observed", xlabel="Ω (π/sec)", ylabel="population (right)")
plot!(Ω_vals / (π/sec), sagnac_population(Φ, Ω_vals); linestyle=:dash, label="Sagnac", xlabel="Ω (π/sec)", ylabel="population (right)")

# ## Harmonic non-adiabatic evolution

# +
omega_ramp_up_nonadiabatic(t; w0=OMEGA_TARGET, t_r=150μs) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down_nonadiabatic(t; w0=OMEGA_TARGET, t_r=150μs) = w0 * cos(π * t / (2t_r))^2;

args_nonadiabatic = Dict{Symbol,Any}(
    :theta_grid => theta_grid,
    :potential_depth => 0.2MHz,
    :omega_up => omega_ramp_up_nonadiabatic,
    :omega_down => omega_ramp_down_nonadiabatic,
    :omega_0 => OMEGA_TARGET,
    :t_r => 150μs,
    :n_cycles => 10,
    :nt_free => 10000,
    :model => :harmonic,
)

# +
frame=:lab

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args_nonadiabatic...,
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

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args_nonadiabatic...,
    parallel=true,
    ret=:expvals,
    frame,
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left, expvals_right;
    show_standard_deviation=false,
    frame
)
# -

# Compare these with the expected oscillation period:

ωₕ = sqrt(0.2MHz  * N_SITES^2 / EFFECTIVE_MASS)

# This corresponds to oscillations on the time scale of:

(2π / ωₕ) / ms

plot(tlists[2] ./ ms, expvals_left[2][1,:] .- 0.125π)
plot!(tlists[2] ./ ms, 0.004 .* sin.(ωₕ .* tlists[2]))

# Zooming in on the first two ms:

plot!(;xlim=(0,2))

function get_spectrum(signal, tlist)
    dt = tlist[2] - tlist[1]
    N = length(tlist)
    fs = 2π / dt
    freqs = collect(fftshift(fftfreq(N, fs)))
    spectrum = abs2.(fftshift(fft(signal)))
    return freqs, spectrum
end

θ_freqs, θ_spectrum = get_spectrum(expvals_left[2][1,:] .- 0.125π, tlists[2]);
#θ_freqs, θ_spectrum = get_spectrum(sin.(ωₕ .* tlists[2]), tlists[2]);

plot(θ_freqs, θ_spectrum; marker=true, label="spectrum")
vline!([ωₕ,]; label="ωₕ", xlim=(0, 2ωₕ))

# +
frame=:mixed

tlists, omega_vals, expvals_left, expvals_right = propagate_scheme(;
    args_nonadiabatic...,
    parallel=true,
    ret=:expvals,
    frame,
);

plot_full_pos_mom_dynamics(
    tlists,
    expvals_left, expvals_right;
    show_standard_deviation=false,
    frame
)
# -


