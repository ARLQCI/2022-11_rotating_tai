using FFTW
import QuantumPropagators.Storage: map_observables

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


"""Calculate values `[⟨θ⟩, σ_θ, ⟨p⟩, σ_p]`.

The values `σ_θ` and `σ_p` are the standard deviations from the expectation
values ⟨θ⟩ and ⟨p⟩
"""
function map_observables(observables::PositionMomentumObservables, Ψ)
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
