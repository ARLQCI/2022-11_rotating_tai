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

# # Analyze the Separation Dynamics for Different Points in the Landscape

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
const MOMENTUM_TARGET = - EFFECTIVE_MASS * OMEGA_TARGET;

# ## Map of Analysis points

potential_depth_values = collect(range(0.1MHz, 2.2MHz, length=106))
potential_depth_values ./ MHz;

separation_time_orders_of_magnitude = collect(range(-1, 5, length=121));

separation_time_values = [10^x * μs for x in separation_time_orders_of_magnitude];

import FileIO

map_data = FileIO.load("./data/2023-01-05_map_splitting_fidelity.npz");

POINTS = [
    (2.1MHz,
        [
            1e-1 * sec,
            5e-4 * sec,
            3e-4 * sec,
            2e-4 * sec,
            1.5e-4 * sec,
            1e-4 * sec,
            1e-5 * sec,
            1e-7 * sec,
        ]
    )
    (0.2MHz,
        [
            1e-1 * sec,
            5e-4 * sec,
            3e-4 * sec,
            2e-4 * sec,
            1.5e-4 * sec,
            1e-4 * sec,
            1e-5 * sec,
            1e-7 * sec,
        ]
    )
]

# +
function plot_points()
    fig = contourf(
        separation_time_values ./ sec,
        potential_depth_values ./ MHz,
        map_data,
        tick_direction=:out,
        xminorticks=9,
        yminorticks=2,
        xaxis=:log10,
        ylabel="V₀ (MHz)",
        xlabel="separation time (seconds)",
        title=raw"Separation Fidelity $|⟨Ψ(t_r) | Ψ_{\textrm{tgt}}⟩|^2$",
        xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    for (V0, t_r_vals) in POINTS
        scatter!(
            [(t_r ./ sec , V0 ./ MHz) for t_r in t_r_vals],
            color="green",
            label="",
            size=(800, 600),
            series_annotations = text.(1:length(t_r_vals), :bottom)
        )
    end
    fig
end

plot_points()
# -

# ## Propagation

using QuantumPropagators.Controls: evaluate, discretize, discretize_on_midpoints

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;

# **TODO: am I going left or right?**

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

function propagate_splitting(;
    separation_time,
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
    nt = choose_timesteps(separation_time; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, separation_time, length=nt))
    ω_func(t) = omega_ramp_up(t; w0=omega_target, t_r=separation_time)
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

includet("./include/position_momentum_observables.jl")

# +
function get_expval_dynamics(;
    separation_time,
    potential_depth,
    theta_max=0.25π,
    theta_steps=1024,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    show=true,
    show_standard_deviations=false,
    title="θ and p expectation values",
    θ₀=0.125π,
    momentum_target=MOMENTUM_TARGET,
    kwargs...
)

    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    observables = PositionMomentumObservables(; theta_grid=θ)
    expvals = propagate_splitting(;
        separation_time,
        potential_depth,
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
        plot_expval_dynamics(tlist, expvals; show_standard_deviations, title, θ₀, momentum_target)
    else
        return expvals
    end

end
# -

function plot_expval_dynamics(
    tlist, expvals;
    θ₀=0.125π,
    momentum_target=MOMENTUM_TARGET,
    show_standard_deviations=false,
    figsize=(900, 350),
    title="θ and p expectation values",
    margin=15,
    relative_to_theta=zeros(length(tlist)),
    show_lab_frame_displacement=true,
)
    θ = @view expvals[1, :]
    σ_θ = @view expvals[2, :]
    p = @view expvals[3, :]
    σ_p = @view expvals[4, :]
    θ′ = θ .- θ₀ .- relative_to_theta
    if show_standard_deviations
        ax_pos =
            plot(tlist ./ sec, θ′ ./ π; ribbon=σ_θ ./ π, label="", xlabel="time", ylabel="Δθ (π)")
        ax_mom = plot(tlist ./ sec, p; ribbon=σ_p, label="p(t)", xlabel="time", ylabel="momentum")
    else
        ax_pos = plot(tlist ./ sec, θ′ ./ π; label="", xlabel="time", ylabel="Δθ (π)")
        ax_mom = plot(tlist ./ sec, p; label="p(t)", xlabel="time", ylabel="momentum")
    end
    if show_lab_frame_displacement
        separation_time = tlist[end]
        displacement = lab_frame_displacement(tlist, separation_time)[end]
        hline!(ax_pos, [displacement / π ], ls=:dash, label="")
    end
    hline!(ax_mom, [momentum_target,], color="black", ls=:dash, label="target")
    plot(ax_pos, ax_mom; size=figsize, plot_title=title, margin=(margin * Plots.px))
end

# ## 2.1 MHz

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

# +
function lab_frame_displacement(tlist::Vector{Float64}, separation_time::Float64; ω₀=OMEGA_TARGET)

    dt = tlist[2] - tlist[1]
    ω = discretize(t -> omega_ramp_up(t; w0=ω₀, t_r=separation_time), tlist)
    @assert ω[1] ≈ 0.0
    θ = cumsum(ω) .* dt
    return θ

end
# -

function scan_t_r(potential_depth_index; frame=:moving, kwargs...)
    i = potential_depth_index
    V0 = POINTS[i][1]
    for (j, t_r) in enumerate(POINTS[i][2])
        title = "($j): V₀ = $(V0 / MHz) MHz, separation time = $(fmt_exp_unicode(t_r / sec)) s ($frame frame)"
        nt = choose_timesteps(t_r)
        tlist = collect(range(0, t_r, length=nt))
        θ_lab = lab_frame_displacement(tlist, t_r) # XXX ???
        @assert length(θ_lab) == length(tlist)
        expvals = get_expval_dynamics(separation_time=t_r, potential_depth=V0, show=false)
        if frame == :moving
            display(plot_expval_dynamics(tlist, expvals; title, kwargs...))
        elseif frame == :lab
            display(plot_expval_dynamics(tlist, expvals; title, relative_to_theta=θ_lab, kwargs...))
        else
            error("Invalid frame=$(repr(frame))")
        end
        println("Lab frame θ displacement: $(fmt_exp_unicode(θ_lab[end] / π))π")
    end
end

scan_t_r(1; frame=:moving, show_standard_deviations=false)

scan_t_r(1; frame=:moving, show_standard_deviations=true)

# ## 0.2 MHz

scan_t_r(2; frame=:moving)

# ## Limit of short separation time

# We see that the for very short separation times, the initial wave packet does not move. Thus, the behavior in the left half of the "map" plot just has to be the overlap of the initial state with the target state, which seems to increase with larger potential depths.

function get_target_overlaps(potential_depth_values; separation_time=1e-6sec)
    overlaps = zeros(length(potential_depth_values))
    for (i, V₀) in enumerate(potential_depth_values)
        Ψ₀, θ = propagate_splitting(;
            potential_depth=V₀, separation_time, ret=:initial_state
        )
        Ψtgt, θ = propagate_splitting(;
            potential_depth=V₀, separation_time, ret=:target
        )
        overlaps[i] = abs2(Ψ₀ ⋅ Ψtgt)
    end
    return overlaps
end

target_overlaps = get_target_overlaps(potential_depth_values);

plot(
    potential_depth_values ./ MHz, target_overlaps;
    label="",
    title="Overlap of initial state with target state",
    xlabel="potential depth (MHz)",
    ylabel="|⟨Ψ₀|Ψtgt⟩|²",
)

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
    kwargs...
)
    fig = plot(; xlabel="momentum", ylabel="amplitude", kwargs...)
    for (i, Ψ) in enumerate(states)
        Ψ_momentum, momentum_grid = psi_to_momentum(Ψ, θ)
        plot!(fig, momentum_grid, abs2.(Ψ_momentum), label=labels[i])
    end
    fig
end

function compare_initial_target_state(;potential_depth, separation_time=1e-6sec, kwargs...)
    Ψ₀, θ = propagate_splitting(;
        potential_depth, separation_time, ret=:initial_state
    )
    Ψtgt, θ = propagate_splitting(;
        potential_depth, separation_time, ret=:target
    )
    fig = plot_states_in_momentum_space(
        θ, Ψ₀, Ψtgt;
        labels=["Ψ₀", "Ψtgt"],
        title="V₀=$(potential_depth/MHz) MHz",
        kwargs...
    )
    display(fig)
    println("Overlap = $(abs2(Ψ₀ ⋅ Ψtgt))")
end

compare_initial_target_state(potential_depth=0.2MHz; xlim=(-500, 500))

compare_initial_target_state(potential_depth=2.2MHz; xlim=(-500, 500))

# Deeper potentials means narrower wave functions in coordinate space, which means broader wave functions in momentum space, which have more overlap.
