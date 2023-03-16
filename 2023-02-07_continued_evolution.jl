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
#     display_name: Julia 1.8 (auto threads)
#     language: julia
#     name: julia-1.8-multithread
# ---

# # Observing the time evolution after the splitting

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
const MOMENTUM_TARGET = -EFFECTIVE_MASS * OMEGA_TARGET;

projectdir(args...) = joinpath(@__DIR__, args...)
plotsdir(args...) = projectdir("plots", "2023-02-07_continued_evolution", args...)
# rm(plotsdir(); recursive=true, force=true)
mkpath(plotsdir())

# ## Prerequisites

using QuantumPropagators.Controls: evaluate, discretize, discretize_on_midpoints

includet("./include/rotating_tai.jl");

includet("./include/split_propagator.jl")

omega_ramp_up(t; w0=OMEGA_TARGET, t_r) = w0 * sin(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

# +
function lab_frame_displacement(
    tlist::Vector{Float64},
    separation_time::Float64;
    ω₀=OMEGA_TARGET,
    ω=discretize(t -> omega_ramp_up(t; w0=ω₀, t_r=separation_time), tlist)
)

    dt = tlist[2] - tlist[1]
    @assert ω[1] ≈ 0.0
    θ = cumsum(ω) .* dt
    return θ

end
# -

# Let's observe the potential moving at a constant OMEGA_TARGET for 1 ms

# +
function plot_lab_frame_moving_potential()

    V₀ = 2.2MHz
    separation_time = 1ms
    tlist = collect(range(0, separation_time, length=1001))
    θ = collect(range(0, 0.25π, length=1024))
    ϕ = discretize_on_midpoints(t -> OMEGA_TARGET * t, tlist)
    m = N_SITES

    anim = @animate for n = 1:5:length(tlist)-1
        V = @. V₀ * cos(m * (θ + ϕ[n]))
        plot(θ ./ π, V ./ MHz, xlabel="θ (π)", ylabel="potential (MHz)", label="")
        vline!([0.125], color="black", ls=:dash, label="", title="t=$(tlist[n]/μs) μs")
    end

    gif(anim, "anim.gif", fps=30)

end

# -

plot_lab_frame_moving_potential()

#  We can see that a positive $\phi$ moves the potential to the left

# ## Map of Analysis points

potential_depth_values = collect(range(0.1MHz, 2.2MHz, length=106))
potential_depth_values ./ MHz;

separation_time_orders_of_magnitude = collect(range(-1, 5, length=121));

separation_time_values = [10^x * μs for x in separation_time_orders_of_magnitude];

import FileIO

map_data = FileIO.load("./data/2023-01-05_map_splitting_fidelity.npz");

POINTS = [
    (
        2.1MHz,
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
    (
        0.2MHz,
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
            [(t_r ./ sec, V0 ./ MHz) for t_r in t_r_vals],
            color="green",
            label="",
            size=(800, 600),
            series_annotations=text.(1:length(t_r_vals), :bottom)
        )
    end
    fig
end

plot_points()
# -

# ## Propagation

omega_ramp_up(t; w0=OMEGA_TARGET, t_r=SEPARATION_TIME) = w0 * sin(π * t / (2t_r))^2;

function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end

function propagate_splitting(;
    separation_time,
    evolution_time,
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
    T = separation_time + evolution_time
    nt = choose_timesteps(T; timesteps_per_microsec, minimum_timesteps)
    tlist = collect(range(0, T, length=nt))


    function ω_func(t)
        if t <= separation_time
            omega_ramp_up(t; w0=omega_target, t_r=separation_time)
        else
            omega_target
        end
    end

    if ret == :omega
        return ω_func
    end

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

using DrWatson: @dict

# +
function get_expval_dynamics(;
    separation_time,
    evolution_time,
    potential_depth,
    theta_max=0.25π,
    theta_steps=1024,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    set_of_observables=:xp,
    kwargs...
)

    θ::Vector{Float64} = collect(range(0, theta_max, length=theta_steps))
    args = Dict{Symbol,Any}(@dict(
        separation_time,
        evolution_time,
        potential_depth,
        theta_max,
        theta_steps,
        timesteps_per_microsec,
        minimum_timesteps,
        storage = true,
    ))
    if set_of_observables == :xp
        observables = PositionMomentumObservables(; theta_grid=θ)
    elseif set_of_observables == :target
        Ψtgt, _ = propagate_splitting(; ret=:target, args..., kwargs...)
        observables = [Ψ -> abs2(Ψ ⋅ Ψtgt), ]
    else
        error("unknown set_of_observables")
    end
    args[:observables] = observables
    expvals = propagate_splitting(; ret=:propagation, args..., kwargs...)
    _, tlist = propagate_splitting(; ret=:system, args...)

    return tlist, expvals

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
    mark_momentum=[MOMENTUM_TARGET],
    mark_momentum_labels=["target"],
    mark_displacements=[],
    mark_times=[]
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
    if length(mark_displacements) > 0
        hline!(ax_pos, mark_displacements, ls=:dash, label="")
    end
    if length(mark_momentum) > 0
        hline!(ax_mom, mark_momentum, color="black", ls=:dash, label=mark_momentum_labels)
    end
    if length(mark_times) > 0
        vline!(ax_pos, mark_times ./ sec, color="black", ls=:dash, label="")
        vline!(ax_mom, mark_times ./ sec, color="black", ls=:dash, label="")
    end
    plot(ax_pos, ax_mom; size=figsize, plot_title=title, margin=(margin * Plots.px))
end

function plot_target_overlap_dynamics(
    tlist,
    expvals;
    title="overlap with target",
    mark_times=[]
)
    fig = plot(tlist ./ sec, expvals; label="", xlabel="time", ylabel="Fidelity", title)
    if length(mark_times) > 0
        vline!(fig, mark_times ./ sec, color="black", ls=:dash, label="")
    end
    return fig
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

function add_suffix(filename, suffix)
    basename, ext = splitext(filename)
    return "$basename$suffix$ext"
end

using FileIO

function scan_t_r(
    potential_depth_index;
    set_of_observables=:xp,
    evolution_time_ratio=1.0,
    frame=:moving,
    outfile=nothing,
    margin=15,
    figsize=(900, 350),
    kwargs...
)
    i = potential_depth_index
    V0 = POINTS[i][1]
    outfiles = []
    for (j, t_r) in enumerate(POINTS[i][2])
        title = "($j): V₀ = $(V0 / MHz) MHz, separation time = $(fmt_exp_unicode(t_r / sec)) s ($frame frame)"
        evolution_time = evolution_time_ratio * t_r
        tlist, expvals = get_expval_dynamics(;
            set_of_observables,
            evolution_time,
            separation_time=t_r,
            potential_depth=V0,
        )
        θ_lab = lab_frame_displacement(tlist, t_r) # XXX ???
        @assert length(θ_lab) == length(tlist)
        if frame == :moving
            if set_of_observables == :xp
                fig = plot_expval_dynamics(
                    tlist,
                    expvals;
                    title,
                    mark_times=(evolution_time > 0.0 ? [t_r,] : []),
                    figsize,
                    margin,
                    kwargs...
                )
            elseif set_of_observables == :target
                fig = plot_target_overlap_dynamics(
                    tlist,
                    expvals;
                    title,
                    mark_times=(evolution_time > 0.0 ? [t_r,] : []),
                )
            else
                error("unkown set_of_observables")
            end
        else
            error("Invalid frame=$(repr(frame))")
        end
        if isnothing(outfile)
            display(fig)
        else
            outfile_j = add_suffix(outfile, "_$j")
            savefig(fig, outfile_j)
            push!(outfiles, outfile_j)
            println("Written plot to $outfile_j")
        end
        println("($j) Lab frame θ displacement: $(fmt_exp_unicode(θ_lab[end] / π))π")
    end
    if !isnothing(outfile)
        n = length(outfile)
        imgs = [load(file) for file in outfiles]
        save(outfile, vcat(imgs...))
        println("Written combined plot to $outfile")
    end
end

scan_t_r(
    1;
    evolution_time_ratio=10.0,
    frame=:moving,
    show_standard_deviations=false,
    outfile=plotsdir("V0_1.png")
)

# ![](plots/2023-02-07_continued_evolution/V0_1.png)

scan_t_r(
    1;
    evolution_time_ratio=10.0,
    frame=:moving,
    show_standard_deviations=true,
    outfile=plotsdir("V0_1_with_sd.png")
)

# ![](plots/2023-02-07_continued_evolution/V0_1_with_sd.png)

scan_t_r(
    1;
    set_of_observables=:target,
    evolution_time_ratio=10.0,
    frame=:moving,
    show_standard_deviations=false
)

# ## 0.2 MHz

scan_t_r(2; frame=:moving, outfile=plotsdir("V0_2.png"))

# ![](plots/2023-02-07_continued_evolution/V0_2.png)
