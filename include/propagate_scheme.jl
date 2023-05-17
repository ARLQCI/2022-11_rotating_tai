using QuantumPropagators
using QuantumPropagators.Controls: discretize, discretize_on_midpoints
using QuantumPropagators.Storage: init_storage
using LinearAlgebra
using DataFrames
using FFTW
using Test
import QuantumPropagators.Storage: map_observables


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


function rotating_tai_hamiltonian_coord(;
    tlist=nothing,  # not used
    theta_grid,
    ω,
    scale_potential=nothing,  # nothing, or function of time
    potential_depth,
    number_of_sites,
    mass,
    Ω=0.0,
    model=:cos,
    direction=1
)

    m = number_of_sites
    V₀ = potential_depth
    θ = theta_grid

    if model == :cos
        V = Diagonal(V₀ .* cos.(m .* θ))
    elseif model == :harmonic
        V = let θ₀ = π / number_of_sites, m = number_of_sites, M = mass
            ω₀ = sqrt(V₀ * m^2 / M)
            Diagonal((0.5 * M * ω₀^2) .* (θ .- θ₀).^2)
        end
    else
        error("Invalid model: $(repr(model))")
    end
    if !isnothing(scale_potential)
        V = hamiltonian((V, scale_potential); check=false)
    end

    dθ = θ[2] - θ[1]
    nθ = length(θ)
    P = p_matrix(θ)
    K = P^2 / (2 * mass)

    K′ = K - Ω * P

    if ω isa Number
        if direction == 1
            H = K′ + ω * P + V
        elseif direction == -1
            H = K′ - ω * P + V
        else
            error("direction must be ±1")
        end
    else
        if direction == 1
            H = hamiltonian(K′ + V, (P, ω))
        elseif direction == -1
            H = hamiltonian(K′ + V, (-P, ω))
        else
            error("direction must be ±1")
        end
    end
    return H
end


function choose_timesteps(separation_time; timesteps_per_microsec=1, minimum_timesteps=1001)
    return max(minimum_timesteps, Int(separation_time ÷ μs) * timesteps_per_microsec + 1)
end


@doc raw"""
Evaluate a cumulative integral consistent with pulse discretization.

```julia
ϕ = cumint(ω, tlist; start=0.0)
```

calculates the array `ϕ` with elements

```math
ϕ(t_i) = ϕ_0 + \int_{t_1}^{t_i} ω(t) dt
```

where ``t_i`` is the i'th element of `tlist` and ``ϕ_0 = ϕ(t_1)`` is the value
of `start`.

For a uniform time grid with step size `dt`, the discretization is such that
that

```
cumint(ω, tlist)[end]  == sum(discretize_on_midpoints(ω, tlist)) * dt
```
"""
function cumint(ω, tlist::Vector{Float64}; start=0.0)
    ω_vals::Vector = discretize_on_midpoints(ω, tlist)
    ϕ = Vector{Float64}(undef, length(tlist))
    ϕ[1] = start
    for i = 1:length(ω_vals)
        ϕ[i+1] = ϕ[i] + ω_vals[i] * (tlist[i+1] - tlist[i])
    end
    return ϕ
end


function test_cumint()
    tlist = collect(range(0, 20; length=100))
    ω = rand(length(tlist))
    ϵ = discretize_on_midpoints(ω, tlist)
    dt = tlist[2] - tlist[1]
    @test isapprox(cumint(ω, tlist)[end], sum(ϵ) * dt; atol=1e-12)
end


"""Propagate an entire pulse scheme for a single wave function.

# Positional Arguments

* `Ψ₀`: the initial wave function
* `θ`: in theta grid on which `Ψ₀` is defined

# Keyword Arguments

* `omega_up`: The control to use for the "up" part of the scheme. Can be a
  function of t ∈ [0, t_r] or a vector appropriate to the "up" time grid of
  length `choose_timesteps(t_r; …)`
* `omega_down`: The control to use for the "down" part of the scheme. Can be a
  function of t ∈ [0, t_r] (!!!) or a vector appropriate to the "down" time
  grid of length `choose_timesteps(t_r; …)`
* `omega_0`: The value of ω₀ to use during the free time evolution
* `t_r`: The time for the "up" and "down" segments
* `direction`: either 1 for "left" or -1 for "right" (sign of ω(t))
* `n_cycles`: How many loops to do in the free time evolution (multiples of π)
* `nt_free=2`: How many times steps to do during the free time evolution.
* `potential_depth`: Depth of the trapping potential
* `mass=EFFECTIVE_MASS`: Effective mass of the atom
* `number_of_sites=N_SITES`: Number of traps in full circle
* `observables=nothing`: If not `nothing`, a three-tuple of observables for the
  "up", "free", and "down" segments.
* `storages=nothing`: If not `nothing`, a tuple of three storage arrays where
  the propagation should store data from the three segments
* `method=:splitprop`: The propagation method to use for the "up" and "down"
  segment
* `method_free=:freeprop`: The propagation method to use for the "free" segment
* `timesteps_per_microsec=1`: Argument for `choose_timesteps` for the "up" and
  "down" segments
* `minimum_timesteps=1001`: Argument for `choose_timesteps` for the "up" and
  "down" segments
* `Ω=0.0`: background rotation (to be measured)
* `model=:cos`: The potential to use. By default (`model=:cos`), use the cosine
  potential. Alternatively, for `model=:harmonic`, use a harmonic approximation
  of the cosine potential.
* `ret=:state`: What to return. One of :state (final state), :tlist (time grids
  for the three segments), :omega_vals (values of `omega_up`, `omega_0`, and
  `omega_down` at the points of the respective time grid), discre:ham
  (Hamiltonians "up", "free", "down")

All other kwargs are passed on to the `propagate` routine for all three
segments.
"""
function propagate_scheme_one_direction(
    Ψ₀,
    θ;
    omega_up,
    omega_down,
    omega_0,
    t_r,
    direction,
    n_cycles,
    nt_free=2,
    potential_depth,
    mass=EFFECTIVE_MASS,
    number_of_sites=N_SITES,
    observables=nothing,
    storages=nothing,
    method=:splitprop,
    method_free=:freeprop,
    timesteps_per_microsec=1,
    minimum_timesteps=1001,
    Ω=0.0,
    model=:cos,
    ret=:state,
    kwargs...
)
    @assert length(θ) == length(Ψ₀)
    @assert (direction == 1) || (direction == -1)

    # Time grids and discretized controls

    # "up" segment
    nt = choose_timesteps(t_r; timesteps_per_microsec, minimum_timesteps)
    tlist_up = collect(range(0, t_r, length=nt))
    dt = tlist_up[2] - tlist_up[1]
    ω_up = discretize_on_midpoints(omega_up, tlist_up)
    ω₀ = omega_0
    @assert ω_up[end] ≈ ω₀
    Φ = sum(ω_up) * dt # Accumulated movement

    # "down" segment
    tlist_down = collect(range(0, t_r, length=nt))  # temporarily, see below
    ω_down = discretize_on_midpoints(omega_down, tlist_down)
    @assert ω_down[1] ≈ ω₀
    Φ += sum(ω_down) * dt  # Accumulated movement

    # "free" segment (requires knowledge of Φ from "up" and "down" segments)
    Φtgt = n_cycles * π
    t_free = (Φtgt - Φ) / ω₀
    if t_free ≤ 0.0
        @error "No free time evolution: 'up' and 'down' already achieve $(Φ/π) cycles"
    end
    if nt_free ≤ 2
        tlist_free = [t_r, t_r + t_free]
    else
        tlist_free = collect(range(t_r, t_r + t_free, length=nt_free))
    end

    # "down" segment (corrected time grid)
    tlist_down = collect(range(t_r + t_free, 2 * t_r + t_free, length=nt))

    if ret == :tlist
        return (tlist_up, tlist_free, tlist_down)
    end

    if ret == :omega_vals
        return (
            discretize(ω_up, tlist_up),
            Float64[ω₀ for t in tlist_free],
            discretize(ω_down, tlist_down)
        )
    end

    # Hamiltonians

    Ĥ_up = rotating_tai_hamiltonian(;
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
        ω=ω_up,
        Ω,
        model,
        direction
    )

    Ĥ_free = rotating_tai_hamiltonian_coord(;
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
        ω=ω₀,
        Ω,
        model,
        direction
    )

    Ĥ_down = rotating_tai_hamiltonian(;
        potential_depth,
        theta_grid=θ,
        mass,
        number_of_sites,
        ω=ω_down,
        Ω,
        model,
        direction
    )

    if ret == :ham
        return (Ĥ_up, Ĥ_free, Ĥ_down)
    end

    kwargs_up = Dict{Symbol,Any}()
    kwargs_free = Dict{Symbol,Any}()
    kwargs_down = Dict{Symbol,Any}()
    if storages ≢ nothing
        kwargs_up[:storage] = storages[1]
        kwargs_free[:storage] = storages[2]
        kwargs_down[:storage] = storages[3]
    end
    if observables ≢ nothing
        kwargs_up[:observables] = observables[1]
        kwargs_free[:observables] = observables[2]
        kwargs_down[:observables] = observables[3]
    end

    #! format: off
    Ψ = propagate(
        Ψ₀,
        Ĥ_up,
        tlist_up;
        method,
        kwargs_up...,
        kwargs...
    )
    Ψ = propagate(
        Ψ,
        Ĥ_free,
        tlist_free;
        method=method_free,
        kwargs_free...,
        kwargs...
    )
    Ψ = propagate(
        Ψ,
        Ĥ_down,
        tlist_down;
        method,
        kwargs_down...,
        kwargs...
    )
    #! format: on

    if ret == :state
        return Ψ
    else
        error("Invalid ret=$ret")
    end

end


function psi_mixed_to_moving_frame(
    Ψ::Vector{ComplexF64},
    θ::Vector{Float64},
    ω::Float64;
    mass=EFFECTIVE_MASS
)
    c = 1im * ω * mass
    θ̂ = Diagonal(θ)
    return exp(c * θ̂) * Ψ
end


"""
Wrapper around existing `observables` that transforms the state momentum to
the moving frame before evaluation.
"""
struct MovingMomentumFrameObservablesWrapper
    observables
    omega::Vector{Float64}
    mass::Float64
    theta_grid::Vector{Float64}
    moving_frame_state::Vector{ComplexF64}
end


function map_observables(observables::MovingMomentumFrameObservablesWrapper, tlist, i, Ψ)
    ω = observables.omega[i]
    c = 1im * ω * observables.mass
    θ = observables.theta_grid
    Ψ′ = observables.moving_frame_state
    @. Ψ′ = exp(c * θ) * Ψ
    return map_observables(observables.observables, tlist, i, Ψ′)
end




"""Propagate an entire pulse scheme from scratch.

# Keyword arguments

* `theta_grid`: the grid for θ. Defaults to 1024 grid point ∈ [0, 0.25π].
* `theta_steps=1024`:, the number of θ grid points (= dimension of Hilbert space)
* `number_of_sites=N_SITES`: Number of traps in full circle
* `potential_depth`: Depth of the trapping potential
* `omega_up`: The control to use for the "up" part of the scheme. Can be a
  function of t ∈ [0, t_r] or a vector appropriate to the "up" time grid of
  length `choose_timesteps(t_r)`
* `omega_down`: The control to use for the "down" part of the scheme. Can be a
  function of t ∈ [0, t_r] (!!!) or a vector appropriate to the "down" time
  grid of length `choose_timesteps(t_r)`
* `omega_0`: The value of ω₀ to use during the free time evolution
* `t_r`: The time for the "up" and "down" segments
* `n_cycles`: How many loops to do in the free time evolution (multiples of π)
* `nt_free=2`: How many times steps to do during the free time evolution.
* `Ω=0.0`: background rotation (to be measured)
* `parallel=true`: Whether to propagate the left and right direction in parallel
* `method=:splitprop`: The propagation method to use for the "up" and "down"
  segment
* `method_free=:freeprop`: The propagation method to use for the "free" segment
* `model=:cos`: The potential to use. By default (`model=:cos`), use the cosine
  potential. Alternatively, for `model=:harmonic`, use a harmonic approximation
  of the cosine potential.
* `initialize_with_Ω=true`: Whether to include the background rotation `Ω` when
  calculating the initial eigenstate.
* `frame=:mixed`: The frame for the returned states or expectation values.
  One of:
  - `:lab`: Both position and momentum are in the absolute lab frame
  - `:mixed`: Position in the moving frame, momentum is in the lab frame. This
    is the frame in which the numerical Hamiltonian is defined, and hence the
    default.
  - `:moving`: Both position and momentum are in the moving frame
* `ret=:P_right`: What to return, one of:
  - `:P_right`: population in the "right" potential
  - `:initial_state`: The initial state (single surface)
  - `:final_states`: the final-time states `Ψ_left, Ψ_right`
  - `:states`: tuple `tlists`, `omega_vals`, `storages_left`, `storages_right`,
    where each element is another 3-tuple for the "up", "free" and "down"
    segments
  - `:expvals`: like `:states`, but with the expectation values for position
    and momentum  in the storages
  - `:omega_vals`: tuple `tlists`, `omega_vals`, where each element is another
    3-tuple fo the "up", "free", and "down" segments. The `omega_vals` are the
    values of ω for each point on the time grid for that segment.

All other kwargs are passed on to the `propagate` routine for all three
segments and for both the "left" and "right" wave function
"""
function propagate_scheme(;
    theta_grid=collect(range(0, 0.25π, length=1024)),
    number_of_sites=N_SITES,
    potential_depth,
    mass=EFFECTIVE_MASS,
    omega_up,
    omega_down,
    omega_0,
    t_r,
    n_cycles=1,
    nt_free=2,
    Ω=0.0,
    parallel=true,
    method=:splitprop,
    method_free=:freeprop,
    model=:cos,
    frame=:mixed,
    ret=:P_right,
    initialize_with_Ω=true,
    kwargs...
)

    θ = theta_grid
    Ψ_empty = zeros(ComplexF64, length(θ))

    allowed_frames = [:lab, :mixed, :moving]
    if frame ∉ allowed_frames
        error("frame $(repr(frame)) must be one of $allowed_frames")
    end
    if frame == :lab
        if ret ∈ [:states, :final_states]
            error("Propagated states cannot be represented in the lab frame")
            # We'd need a full θ grid [0, 2π]
        end
    end

    prop_args = Dict{Symbol,Any}(
        :omega_up => omega_up,
        :omega_down => omega_down,
        :omega_0 => omega_0,
        :t_r => t_r,
        :n_cycles => n_cycles,
        :nt_free => nt_free,
        :potential_depth => potential_depth,
        :number_of_sites => number_of_sites,
        :method => method,
        :method_free => method_free,
        :model => model,
        :Ω => 0.0,  # initially, for determining Ψ₀
    )
    if initialize_with_Ω # XXX
        prop_args[:Ω] = Ω
    end
    for (key, val) in kwargs
        prop_args[key] = val
    end

    Ĥ₀_of_t, _, _ =
        propagate_scheme_one_direction(Ψ_empty, θ; ret=:ham, direction=1, prop_args...)

    tlists =
        propagate_scheme_one_direction(Ψ_empty, θ; ret=:tlist, direction=1, prop_args...)

    omega_vals = propagate_scheme_one_direction(
        Ψ_empty,
        θ;
        ret=:omega_vals,
        direction=1,
        prop_args...
    )

    if ret == :omega_vals
        return tlists, omega_vals
    end

    tlist_up, tlist_free, tlist_down = tlists

    Ĥ₀ = evaluate(Ĥ₀_of_t, tlist_up, 1)

    Ψleft = get_ground_state(Ĥ₀, θ, (2π / (2 * number_of_sites)))
    Ψright = zeros(ComplexF64, length(Ψleft))

    if ret == :initial_state
        return Ψleft
    end

    prop_args[:Ω] = Ω
    storages_left = nothing
    storages_right = nothing
    if :observables in keys(kwargs)
        error("Cannot use external `observables`")
    end
    observables_left_up = (Ψ -> Ψ,)
    observables_left_free = (Ψ -> Ψ,)
    observables_left_down = (Ψ -> Ψ,)
    observables_right_up = (Ψ -> Ψ,)
    observables_right_free = (Ψ -> Ψ,)
    observables_right_down = (Ψ -> Ψ,)
    if ret == :expvals
        observables_left_up = PositionMomentumObservables(; theta_grid)
        observables_left_free = PositionMomentumObservables(; theta_grid)
        observables_left_down = PositionMomentumObservables(; theta_grid)
        observables_right_up = PositionMomentumObservables(; theta_grid)
        observables_right_free = PositionMomentumObservables(; theta_grid)
        observables_right_down = PositionMomentumObservables(; theta_grid)
    end
    if frame == :moving
        observables_left_up = MovingMomentumFrameObservablesWrapper(
            observables_left_up,
            omega_vals[1],
            mass,
            θ,
            similar(Ψleft)
        )
        observables_left_free = MovingMomentumFrameObservablesWrapper(
            observables_left_free,
            omega_vals[2],
            mass,
            θ,
            similar(Ψleft)
        )
        observables_left_down = MovingMomentumFrameObservablesWrapper(
            observables_left_down,
            omega_vals[3],
            mass,
            θ,
            similar(Ψleft)
        )
        observables_right_up = MovingMomentumFrameObservablesWrapper(
            observables_right_up,
            -omega_vals[1],
            mass,
            θ,
            similar(Ψright)
        )
        observables_right_free = MovingMomentumFrameObservablesWrapper(
            observables_right_free,
            -omega_vals[2],
            mass,
            θ,
            similar(Ψright)
        )
        observables_right_down = MovingMomentumFrameObservablesWrapper(
            observables_right_down,
            -omega_vals[3],
            mass,
            θ,
            similar(Ψright)
        )
    end
    #! format: off
    observables_left =
        (observables_left_up, observables_left_free, observables_left_down)
    observables_right =
        (observables_right_up, observables_right_free, observables_right_down)
    #! format: on
    if ret ∈ (:states, :expvals)
        storages_left = (
            init_storage(Ψleft, tlist_up, observables_left_up),
            init_storage(Ψleft, tlist_free, observables_left_free),
            init_storage(Ψleft, tlist_down, observables_left_down)
        )
        storages_right = (
            init_storage(Ψright, tlist_up, observables_right_up),
            init_storage(Ψright, tlist_free, observables_right_free),
            init_storage(Ψright, tlist_down, observables_right_down)
        )
    end


    U_πhalf = [
        1  𝕚
        𝕚  1
    ]
    # In principle, the above should be
    #
    #     U_πhalf = [
    #         1/√2  𝕚/√2
    #         𝕚/√2  1/√2
    #     ]
    #
    # but we modify this so that the wave function in each direction is
    # normalized. This make it much easier to calculate expecation values on
    # the two surfaces. We correct this in the final recombination.

    Ψright, Ψleft = U_πhalf * [Ψright, Ψleft]

    if parallel
        prop_left = Threads.@spawn propagate_scheme_one_direction(
            Ψleft,
            θ;
            prop_args...,
            direction=1,
            storages=storages_left,
            observables=observables_left
        )
        prop_right = Threads.@spawn propagate_scheme_one_direction(
            Ψright,
            θ;
            prop_args...,
            direction=-1,
            storages=storages_right,
            observables=observables_right
        )
        Ψleft = fetch(prop_left)
        Ψright = fetch(prop_right)
    else
        Ψleft = propagate_scheme_one_direction(
            Ψleft,
            θ;
            prop_args...,
            direction=1,
            storages=storages_left,
            observables=observables_left
        )
        Ψright = propagate_scheme_one_direction(
            Ψright,
            θ;
            prop_args...,
            direction=-1,
            storages=storages_right,
            observables=observables_right
        )
    end

    if (frame == :lab) && (ret == :expvals)
        # Adjust the θ expectation values to be in the lab frame
        Δθ_lab_up = mod2pi.(cumint(omega_vals[1], tlist_up; start=0.0))
        Δθ_lab_free = mod2pi.(cumint(omega_vals[2], tlist_free; start=Δθ_lab_up[end]))
        Δθ_lab_down = mod2pi.(cumint(omega_vals[3], tlist_down; start=Δθ_lab_free[end]))
        @. storages_left[1][1, :] -= Δθ_lab_up
        @. storages_left[2][1, :] -= Δθ_lab_free
        @. storages_left[3][1, :] -= Δθ_lab_down
        @. storages_right[1][1, :] += Δθ_lab_up
        @. storages_right[2][1, :] += Δθ_lab_free
        @. storages_right[3][1, :] += Δθ_lab_down
    end

    Ψleft, Ψright = (0.5 * U_πhalf) * [Ψleft, Ψright]
    # The factor 0.5 corrects the "modified" U_πhalf. The resulting Ψleft and
    # Ψright are no longer independently normalized. Instead, the square of
    # their norm corresponds to the relative population.

    if ret ∈ (:states, :expvals)
        return tlists, omega_vals, storages_left, storages_right
    elseif ret == :final_states
        if frame == :mixed
            return Ψleft, Ψright
        elseif frame == :moving
            Ψleft = psi_mixed_to_moving_frame(Ψleft, θ, omega_vals[3][end]; mass)
            Ψright = psi_mixed_to_moving_frame(Ψright, θ, -omega_vals[3][end]; mass)
            return Ψleft, Ψright
        elseif frame == :lab
            error("Lab frame final states cannot be represented.")
        else
            error("Invalid frame")
        end
    elseif ret == :P_right
        return norm(Ψright)^2
    else
        error("Invalid ret=$ret")
    end

end


function plot_full_pos_mom_dynamics(
    tlists,
    expvals_left,
    expvals_right;
    mass=EFFECTIVE_MASS,
    time_unit=:ms,
    show_standard_deviation=true,
    θ₀=0.125π,
    relative_to_omega_vals=nothing,
    plot_title="position and momentum expectation values",
    frame=nothing
)
    show_sd = show_standard_deviation
    if frame ≢ nothing
        plot_title *= " ($frame frame)"
    end

    tlist_up, tlist_free, tlist_down = tlists
    expvals_left_up, expvals_left_free, expvals_left_down = expvals_left
    expvals_right_up, expvals_right_free, expvals_right_down = expvals_right
    omega_up = t -> 0.0
    omega_free = t -> 0.0
    omega_down = t -> 0.0
    if relative_to_omega_vals ≢ nothing
        omega_up, omega_free, omega_down = omega_vals
    end

    # position

    θ_left_up = @view expvals_left_up[1, :]
    σθ_left_up = @view expvals_left_up[2, :]
    θ_right_up = @view expvals_right_up[1, :]
    σθ_right_up = @view expvals_right_up[2, :]
    ax_pos_up = plot(
        tlist_up ./ eval(time_unit),
        (θ_left_up .- θ₀) ./ π;
        ribbon=(show_sd ? (σθ_left_up ./ π) : nothing),
        label="left",
        xlabel="time ($time_unit)",
        ylabel="Δθ (π)"
    )
    plot!(
        ax_pos_up,
        tlist_up ./ eval(time_unit),
        (θ_right_up .- θ₀) ./ π;
        ribbon=(show_sd ? (σθ_right_up ./ π) : nothing),
        label="right"
    )

    θ_left_free = @view expvals_left_free[1, :]
    σθ_left_free = @view expvals_left_free[2, :]
    θ_right_free = @view expvals_right_free[1, :]
    σθ_right_free = @view expvals_right_free[2, :]
    ax_pos_free = plot(
        tlist_free ./ eval(time_unit),
        (θ_left_free .- θ₀) ./ π;
        ribbon=(show_sd ? (σθ_left_free ./ π) : nothing),
        label="",
        xlabel="time ($time_unit)",
        yformatter=_ -> ""
    )
    plot!(
        ax_pos_free,
        tlist_free ./ eval(time_unit),
        (θ_right_free .- θ₀) ./ π;
        ribbon=(show_sd ? (σθ_right_free ./ π) : nothing),
        label=""
    )

    θ_left_down = @view expvals_left_down[1, :]
    σθ_left_down = @view expvals_left_down[2, :]
    θ_right_down = @view expvals_right_down[1, :]
    σθ_right_down = @view expvals_right_down[2, :]
    ax_pos_down = plot(
        tlist_down ./ eval(time_unit),
        (θ_left_down .- θ₀) ./ π;
        ribbon=(show_sd ? (σθ_left_down ./ π) : nothing),
        label="",
        xlabel="time ($time_unit)",
        yformatter=_ -> ""
    )
    plot!(
        ax_pos_down,
        tlist_down ./ eval(time_unit),
        (θ_right_down .- θ₀) ./ π;
        ribbon=(show_sd ? (σθ_right_down ./ π) : nothing),
        label=""
    )

    # momentum

    momentum_unit = mass * π / sec
    p_left_up = @view expvals_left_up[3, :]
    σp_left_up = @view expvals_left_up[4, :]
    ω_up = discretize(omega_up, tlist_up)
    p0_left_up = -mass .* ω_up
    p0_right_up = mass .* ω_up
    p_right_up = @view expvals_right_up[3, :]
    σp_right_up = @view expvals_right_up[4, :]
    ax_mom_up = plot(
        tlist_up ./ eval(time_unit),
        (p_left_up .- p0_left_up) ./ momentum_unit;
        ribbon=(show_sd ? (σp_left_up ./ momentum_unit) : nothing),
        label="",
        xlabel="time ($time_unit)",
        ylabel=(isnothing(relative_to_omega_vals) ? "p (M π/sec)" : "Δp (M π/sec)")
    )
    plot!(
        ax_mom_up,
        tlist_up ./ eval(time_unit),
        (p_right_up .- p0_right_up) ./ momentum_unit;
        ribbon=(show_sd ? (σp_right_up ./ momentum_unit) : nothing),
        label=""
    )

    p_left_free = @view expvals_left_free[3, :]
    σp_left_free = @view expvals_left_free[4, :]
    ω_free = discretize(omega_free, tlist_free)
    p0_left_free = -mass .* ω_free
    p0_right_free = mass .* ω_free
    p_right_free = @view expvals_right_free[3, :]
    σp_right_free = @view expvals_right_free[4, :]
    ax_mom_free = plot(
        tlist_free ./ eval(time_unit),
        (p_left_free .- p0_left_free) ./ momentum_unit;
        ribbon=(show_sd ? (σp_left_free ./ momentum_unit) : nothing),
        label="",
        xlabel="time ($time_unit)",
        yformatter=_ -> ""
    )
    plot!(
        ax_mom_free,
        tlist_free ./ eval(time_unit),
        (p_right_free .- p0_right_free) ./ momentum_unit;
        ribbon=(show_sd ? (σp_right_free ./ momentum_unit) : nothing),
        label=""
    )

    p_left_down = @view expvals_left_down[3, :]
    σp_left_down = @view expvals_left_down[4, :]
    ω_down = discretize(omega_down, tlist_down)
    p0_left_down = -mass .* ω_down
    p0_right_down = mass .* ω_down
    p_right_down = @view expvals_right_down[3, :]
    σp_right_down = @view expvals_right_down[4, :]
    ax_mom_down = plot(
        tlist_down ./ eval(time_unit),
        (p_left_down .- p0_left_down) ./ momentum_unit;
        ribbon=(show_sd ? (σp_left_down ./ momentum_unit) : nothing),
        label="",
        xlabel="time ($time_unit)",
        yformatter=_ -> ""
    )
    plot!(
        ax_mom_down,
        tlist_down ./ eval(time_unit),
        (p_right_down .- p0_right_down) ./ momentum_unit;
        ribbon=(show_sd ? (σp_right_down ./ momentum_unit) : nothing),
        label=""
    )

    row1 = plot(ax_pos_up, ax_pos_free, ax_pos_down, layout=(1, 3), link=:y)
    row2 = plot(ax_mom_up, ax_mom_free, ax_mom_down, layout=(1, 3), link=:y)
    plot(row1, row2; layout=(2, 1), size=(1000, 800), plot_title)

end


"""Collect the dyamics data into a datframe.

```julia
df = collect_dynamics_dataframe(
    tlists,
    omega_vals,
    expvals;
    θ₀=0.125π,
    steps_up=1,
    steps_free=1,
    steps_down=1
)
```

where `tlists`, `omega_vals` and `expvals=expvals_left` originate from a call
to `propagate_scheme` with `ret=:expvals` and `expvals`. Note that
`expvals_right` is redundant, since it always mirror `expvals_left`, and only
one of them (`expvals_left` by convention) is passed here as simply `expvals`.

The optional `steps_up`, `steps_free`, and `steps_down` indicate that only
every `steps` data points should be included in the data frame. This is because
the propagation uses orders more points than are necessary for a plot. If the
length of each `tlist` is `10^N+1`, `steps` should be `10^n` with `n<N`.
"""
function collect_dynamics_dataframe(
    tlists,
    omega_vals,
    expvals;
    θ₀=0.125π,
    steps_up=1,
    steps_free=1,
    steps_down=1
)
    DataFrame(
        "time (ms)" => (
            vcat(
                tlists[1][1:steps_up:end-1],
                tlists[2][1:steps_free:end-1],
                tlists[3][1:steps_down:end]
            ) ./ ms
        ),
        "ω (π/sec)" =>
            vcat(
                omega_vals[1][1:steps_up:end-1],
                omega_vals[2][1:steps_free:end-1],
                omega_vals[3][1:steps_down:end]
            ) ./ (π / sec),
        "Δθ (π)" =>
            (
                vcat(
                    expvals[1][1, 1:steps_up:end-1],
                    expvals[2][1, 1:steps_free:end-1],
                    expvals[3][1, 1:steps_down:end]
                ) .- θ₀
            ) ./ π,
        "σ_θ (π)" =>
            (vcat(
                expvals[1][2, 1:steps_up:end-1],
                expvals[2][2, 1:steps_free:end-1],
                expvals[3][2, 1:steps_down:end]
            )) ./ π,
        "p (Mπ/sec)" => (
            vcat(
                expvals[1][3, 1:steps_up:end-1],
                expvals[2][3, 1:steps_free:end-1],
                expvals[3][3, 1:steps_down:end]
            ) ./ (EFFECTIVE_MASS * π / sec)
        ),
        "σ_p (Mπ/sec)" => (
            vcat(
                expvals[1][4, 1:steps_up:end-1],
                expvals[2][4, 1:steps_free:end-1],
                expvals[3][4, 1:steps_down:end]
            ) ./ (EFFECTIVE_MASS * π / sec)
        ),
    )
end


"""Run consistency tests on `propagate_scheme`"""
function test_propagate_scheme()

    theta_grid = collect(range(0, 0.25π, length=1024))
    θ₀ = 0.125π
    pos_mom_obs = PositionMomentumObservables(; theta_grid)

    U_recombine = 0.5 * [
        1  𝕚
        𝕚  1
    ]

    states_left = Dict{Symbol,Any}()
    states_right = Dict{Symbol,Any}()
    expvals_left = Dict{Symbol,Any}()
    expvals_right = Dict{Symbol,Any}()

    prop_args = IdDict{Symbol,Any}(
        :theta_grid => theta_grid,
        :potential_depth => 0.2MHz,
        :omega_up => omega_ramp_up,
        :omega_down => omega_ramp_down,
        :omega_0 => 10π / sec,
        :t_r => 150μs,
        :n_cycles => 2,
        :nt_free => 100,
        :parallel => true,
    )

    local omega_vals

    for frame in [:mixed, :moving, :lab]

        if frame ≠ :lab
            tlists, omega_vals, _states_left, _states_right =
                propagate_scheme(; ret=:states, frame, prop_args...)
            states_left[frame] = _states_left
            states_right[frame] = _states_right
            Ψ_left, Ψ_right = propagate_scheme(; ret=:final_states, frame, prop_args...)
        end

        tlists, omega_vals, _expvals_left, _expvals_right =
            propagate_scheme(; ret=:expvals, frame, prop_args...)
        expvals_left[frame] = _expvals_left
        expvals_right[frame] = _expvals_right

        # end of "up" expvals should match beginning of "free"
        @test norm(expvals_left[frame][1][:, end] - expvals_left[frame][2][:, 1]) < 1e-12
        # end of "free" expvals should match beginning of "down"
        @test norm(expvals_left[frame][2][:, end] - expvals_left[frame][3][:, 1]) < 1e-12
        # "left" should mirror "right"
        for segment in [1, 2, 3]
            # position (should be mirrored)
            @test norm(
                (expvals_left[frame][segment][1, :] .- θ₀) .+
                (expvals_right[frame][segment][1, :] .- θ₀)
            ) < 1e-8
            # position standard deviation (should be the same)
            @test norm(
                expvals_left[frame][segment][2, :] .- expvals_right[frame][segment][2, :]
            ) < 1e-8
            # momentum (should be mirrored)
            @test norm(
                expvals_left[frame][segment][3, :] .+ expvals_right[frame][segment][3, :]
            ) < 1e-8
            # momentum standard deviation (should be the same)
            @test norm(
                expvals_left[frame][segment][4, :] .- expvals_right[frame][segment][4, :]
            ) < 1e-8
        end

        if frame ≠ :lab
            # expectation values should match states
            for segment in [1, 2, 3]
                nt = length(expvals_left[frame][segment][1, :])
                @test norm([
                    norm(
                        map_observables(pos_mom_obs, states_left[frame][segment][:, i]) .-
                        expvals_left[frame][segment][:, i]
                    ) for i = 1:nt
                ]) < 1e-12
                @test norm([
                    norm(
                        map_observables(pos_mom_obs, states_right[frame][segment][:, i]) .-
                        expvals_right[frame][segment][:, i]
                    ) for i = 1:nt
                ]) < 1e-12
            end

            # final states should match recombination of stored states
            Ψ_left_from_storage, Ψ_right_from_storage =
                U_recombine *
                [states_left[frame][3][:, end], states_right[frame][3][:, end]]
            @test norm(Ψ_left .- Ψ_left_from_storage) < 1e-12
            @test norm(Ψ_right .- Ψ_right_from_storage) < 1e-12
        end

    end

    for segment in [1, 2, 3]
        # mixed frame states should match moving frame states
        nt = length(expvals_left[:moving][segment][1, :])
        @test maximum([
            norm(
                states_left[:moving][segment][:, i] - psi_mixed_to_moving_frame(
                    states_left[:mixed][segment][:, i],
                    theta_grid,
                    omega_vals[segment][i]
                )
            ) for i = 1:nt
        ]) < 1e-8
        @test maximum([
            norm(
                states_right[:moving][segment][:, i] - psi_mixed_to_moving_frame(
                    states_right[:mixed][segment][:, i],
                    theta_grid,
                    -omega_vals[segment][i]
                )
            ) for i = 1:nt
        ]) < 1e-8
    end

    return nothing

end
