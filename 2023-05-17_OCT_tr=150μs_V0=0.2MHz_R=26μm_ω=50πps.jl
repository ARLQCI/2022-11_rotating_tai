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

# #  OCT for `t_r=150μs`, `V0=0.2MHz`, `R=26μm`, `ω₀=50π/s`

using QuantumPropagators
using LinearAlgebra
using FFTW
using Serialization
using ProgressMeter

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

const NAME = "2023-05-17_OCT_tr=150μs_V0=0.2MHz_R=26μm_ω=50πps"

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

150μs / sec

# ## Hamiltonian

using QuantumPropagators.Controls: discretize, discretize_on_midpoints

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;
omega_ramp_down(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * cos(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

function propagate_splitting(
    separation_time=SEPARATION_TIME,
    potential_depth=POTENTIAL_DEPTH;
    omega_target=OMEGA_TARGET,
    number_of_sites=N_SITES,
    mass=EFFECTIVE_MASS,
    ret=:fidelity,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    theta_max=0.25π,
    theta_steps=1024,
    kwargs...
)
    nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, separation_time, length=nt))
    ω_func(t) = omega_ramp_up(t; w0=omega_target, t_r=separation_time)
    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    Ĥ = rotating_tai_hamiltonian(;
        tlist,
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
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

HAMILTONIAN, TIME_GRID = propagate_splitting(; ret=:system);

# ## Initial state

INITIAL_STATE, THETA_GRID = propagate_splitting(; ret=:initial_state);

THETA_GRID

# ## Target state

TARGET_STATE, _ = propagate_splitting(; ret=:target);

function psi_to_momentum(Ψ, θ)
    dθ = θ[2] - θ[1]
    nθ = length(θ)
    p::Vector{Float64} = fftshift(2π * fftfreq(nθ, 1 / dθ))
    Ψ_p = fftshift(fft(Ψ))
    return Ψ_p, p
end

function plot_states_in_momentum_space(
    θ,
    states...;
    labels=["Ψ$i" for i = 1:length(states)],
    marker=(-EFFECTIVE_MASS * OMEGA_TARGET),
    marker_label=raw"$Mω_0$",
    kwargs...
)
    fig = plot(; xlabel="momentum", ylabel="amplitude", kwargs...)
    for (i, Ψ) in enumerate(states)
        Ψ_momentum, momentum_grid = psi_to_momentum(Ψ, θ)
        plot!(fig, momentum_grid, abs2.(Ψ_momentum), label=labels[i])
    end
    if !isnothing(marker)
        vline!(fig, [marker], color="black", label=marker_label)
    end
    fig
end

plot_states_in_momentum_space(
    THETA_GRID,
    INITIAL_STATE,
    TARGET_STATE;
    labels=["Ψ₀", "Ψtgt"],
    xlim=(-500, 500)
)

# ## Guess Dynamics

println("Guess fidelity = $(propagate_splitting())")

# +
"""Special object for efficiently calculating expectation values.

See `map_observables`.
"""
struct PositionMomentumObservables

    theta_op::Diagonal{Float64,Vector{Float64}}
    theta_sq_op::Diagonal{Float64,Vector{Float64}}
    momentum_grid::Vector{Float64}
    coordinate_state::Vector{ComplexF64}
    momentum_state::Vector{ComplexF64}
    vals::Vector{Float64}
    FFT
    IFFT

    function PositionMomentumObservables(; theta_grid::Vector{Float64})
        θ = theta_grid
        dθ = θ[2] - θ[1]
        nθ = length(theta_grid)
        momentum_grid::Vector{Float64} = 2π * fftfreq(nθ, 1 / dθ)
        theta_grid_sq = theta_grid .^ 2
        coordinate_state = zeros(ComplexF64, nθ)
        momentum_state = zeros(ComplexF64, nθ)
        vals = zeros(4)
        FFT = plan_fft(coordinate_state; flags=FFTW.MEASURE)
        IFFT = plan_ifft(momentum_state; flags=FFTW.MEASURE)
        new(
            Diagonal(theta_grid),
            Diagonal(theta_grid_sq),
            momentum_grid,
            coordinate_state,
            momentum_state,
            vals,
            FFT,
            IFFT
        )
    end

end
# -

import QuantumPropagators.Storage: map_observables

# +
"""Calculate values `[⟨θ⟩, σ_θ, ⟨p⟩, σ_p]`.

The values `σ_θ` and `σ_p` are the standard deviations from the expectation
values ⟨θ⟩ and ⟨p⟩
"""
function map_observables(observables::PositionMomentumObservables, tlist, i, Ψ)
    # θ expectation value
    exp_val_theta = real(dot(Ψ, observables.theta_op, Ψ))
    exp_val_theta_sq = real(dot(Ψ, observables.theta_sq_op, Ψ))
    variance_theta::Float64 = exp_val_theta_sq - exp_val_theta^2

    # momentum expectation value
    ϕ = observables.coordinate_state
    ϕ̃ = observables.momentum_state
    p = observables.momentum_grid
    mul!(ϕ̃, observables.FFT, Ψ)
    @. ϕ̃ = p * ϕ̃
    mul!(ϕ, observables.IFFT, ϕ̃)
    exp_val_momentum = real(Ψ ⋅ ϕ)
    @. ϕ̃ = p * ϕ̃
    mul!(ϕ, observables.IFFT, ϕ̃)
    exp_val_momentum_sq = real(Ψ ⋅ ϕ)
    variance_momentum::Float64 = exp_val_momentum_sq - exp_val_momentum^2

    observables.vals[1] = exp_val_theta
    observables.vals[2] = sqrt(variance_theta)
    observables.vals[3] = exp_val_momentum
    observables.vals[4] = sqrt(variance_momentum)
    return observables.vals

end

function map_observables(observables::PositionMomentumObservables, Ψ)
    return map_observables(observables, nothing, 1, Ψ)
end
# -

POSITION_MOMENTUM_OBSERVABLES = PositionMomentumObservables(; theta_grid=THETA_GRID);

map_observables(POSITION_MOMENTUM_OBSERVABLES, INITIAL_STATE)

map_observables(POSITION_MOMENTUM_OBSERVABLES, TARGET_STATE)

expval_dynamics = propagate_splitting(;
    observables=POSITION_MOMENTUM_OBSERVABLES,
    storage=true,
    ret=:propagation
)

# +
function get_expval_dynamics(;
    separation_time=SEPARATION_TIME,
    theta_max=0.25π,
    theta_steps=1024,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    show=true,
    show_standard_deviations=false,
    kwargs...
)

    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    observables = PositionMomentumObservables(; theta_grid=θ)
    expvals = propagate_splitting(;
        separation_time,
        theta_max,
        theta_steps,
        timesteps_per_microsec,
        minimum_timesteps,
        observables,
        storage=true,
        ret=:propagation,
        kwargs...
    )

    if show
        nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
        tlist = collect(range(0, separation_time, length=nt))
        plot_expval_dynamics(tlist, expvals; show_standard_deviations)
    else
        return expvals
    end

end
# -

function plot_expval_dynamics(
    tlist,
    expvals;
    θ₀=0.125π,
    momentum_target=MOMENTUM_TARGET,
    show_standard_deviations=false,
    figsize=(900, 350),
    title="θ and p expectation values",
    margin=15,
    relative_to_theta=zeros(length(tlist)),
    show_lab_frame_displacement=true
)
    θ = @view expvals[1, :]
    σ_θ = @view expvals[2, :]
    p = @view expvals[3, :]
    σ_p = @view expvals[4, :]
    θ′ = θ .- θ₀ .- relative_to_theta
    if show_standard_deviations
        ax_pos = plot(
            tlist ./ sec,
            θ′ ./ π;
            ribbon=σ_θ ./ π,
            label="",
            xlabel="time",
            ylabel="Δθ (π)"
        )
        ax_mom = plot(
            tlist ./ sec,
            p;
            ribbon=σ_p,
            label="p(t)",
            xlabel="time",
            ylabel="momentum"
        )
    else
        ax_pos = plot(tlist ./ sec, θ′ ./ π; label="", xlabel="time", ylabel="Δθ (π)")
        ax_mom = plot(tlist ./ sec, p; label="p(t)", xlabel="time", ylabel="momentum")
    end
    if show_lab_frame_displacement
        separation_time = tlist[end]
        displacement = lab_frame_displacement(tlist; separation_time)[end]
        hline!(ax_pos, [displacement / π], ls=:dash, label="")
    end
    hline!(ax_mom, [momentum_target,], color="black", ls=:dash, label="target")
    plot(ax_pos, ax_mom; size=figsize, plot_title=title, margin=(margin * Plots.px))
end

# +
function lab_frame_displacement(
    tlist::Vector{Float64};
    separation_time=SEPARATION_TIME,
    ω₀=OMEGA_TARGET
)

    dt = tlist[2] - tlist[1]
    ω = discretize(t -> omega_ramp_up(t; w0=ω₀, t_r=separation_time), tlist)
    @assert ω[1] ≈ 0.0
    θ = cumsum(ω) .* dt
    return θ

end

# +
using Printf
using Unicode

function superscriptint(i::Int64; _superscripts=collect(graphemes("⁰¹²³⁴⁵⁶⁷⁸⁹")))
    if i < 0
        return "⁻" * superscriptint(-i; _superscripts)
    else
        return join(_superscripts[d+1] for d in reverse(digits(i)))
    end
end

function fmt_exp_unicode(num, fspec=Printf.Format("%.1e"))
    str = Printf.format(fspec, num)
    r_str, e_str = split(str, "e")
    e = parse(Int64, e_str)
    return "$r_str×10$(superscriptint(e))"
end
# -

get_expval_dynamics()

# ## Optimal Control

using QuantumControl

using QuantumControl.Functionals: J_T_sm

includet("./include/guided_amplitude.jl")

function set_guided_control(H, tlist)
    S(t) = QuantumControl.Shapes.flattop(
        t,
        T=tlist[end],
        t_rise=0.2 * tlist[end],
        func=:blackman
    )
    ω_vals::Vector{Float64} = get_controls(H)[1]
    @assert length(ω_vals) == length(tlist) - 1
    control = GuidedAmplitude(t -> 0.0, tlist; guide=omega_ramp_up, shape=S)
    return substitute(H, IdDict(ω_vals => control))
end

objective = Objective(
    initial_state=INITIAL_STATE,
    target_state=TARGET_STATE,
    generator=set_guided_control(HAMILTONIAN, TIME_GRID)
);
δω = get_controls(objective.generator)[1];

problem = ControlProblem(;
    objectives=[objective], tlist=TIME_GRID,
    J_T=J_T_sm, prop_method=:splitprop, verbose=false,
    pulse_options=IdDict(δω => Dict(:lambda_a => 1e6, :update_shape => t -> 1.0)),
    check_convergence=res -> begin ((res.J_T < 1e-8) && (res.converged = true) && (res.message = "J_T < 10⁻⁸")) end
);

res = @optimize_or_load("./data/$NAME.jld2", problem; method=:krotov, iter_stop=400, force=true)

plot(res.guess_controls[1])

plot(res.optimized_controls[1])

H_opt = substitute(
    objective.generator,
    Dict(
        ϵ => discretize_on_midpoints(ϵ_opt, TIME_GRID) for
        (ϵ, ϵ_opt) in zip(get_controls(problem.objectives), res.optimized_controls)
    )
);

function get_amplitudes(H::SplitGenerator)
    amplitudes = H.T.amplitudes
    return Array.(amplitudes)
end

plot(
    TIME_GRID ./ μs,
    discretize(get_amplitudes(H_opt)[1], TIME_GRID) ./ (π / sec),
    linewidth=2,
    label="optimized"
)
plot!(
    TIME_GRID ./ μs,
    discretize(omega_ramp_up, TIME_GRID) ./ (π / sec),
    linewidth=2,
    label="guess"
)
plot!(; xlabel="time (μs)", ylabel="ω (π/sec)")

csv_outfile = "./data/$(NAME)_guess_opt_controls.csv"

open(csv_outfile, "w") do file
  println(file, join(["time (μs)", "ω_guess (π/sec)", "ω_opt (π/sec)"], ","))
  for vals in zip(TIME_GRID ./ μs, discretize(omega_ramp_up, TIME_GRID) ./ (π / sec), discretize(get_amplitudes(H_opt)[1], TIME_GRID) ./ (π / sec))
      println(file, join([@sprintf("%.6f", v) for v in vals], ","))
  end
end

using FileIO: save, load

opt_amplitude_outfile = "./data/$(NAME)_opt_amplitude.npz"

save(
    opt_amplitude_outfile,
    discretize(get_amplitudes(H_opt)[1], TIME_GRID)
);

load(opt_amplitude_outfile) ./ (π/sec)

plot(
    TIME_GRID ./ μs,
    (
        discretize(get_amplitudes(H_opt)[1], TIME_GRID) .-
        discretize(omega_ramp_up, TIME_GRID)
    ) ./ (π / sec),
    label="ΔΩ"
)
plot!(; xlabel="time (μs)", ylabel="ω (π/sec)")

# The solution depends strongly on the value of λₐ
