using LinearAlgebra
using QuantumPropagators
import QuantumPropagators.Controls:
    getcontrols, evalcontrols, evalcontrols!, substitute_controls
import QuantumControlBase: getcontrolderiv


#### Split Operator


# Reminder: any operator needs to implement
#
# * mul!

struct SplitOperator
    T::Diagonal{Float64,Vector{Float64}}
    V::Diagonal{Float64,Vector{Float64}}
    to_p!::Function # coord to momentum
    to_x!::Function # momentum to coord
end


Base.size(O::SplitOperator) = size(O.V)


function LinearAlgebra.mul!(C, A::SplitOperator, B, α, β)
    # |C⟩ = β |C⟩ + α Â |B⟩ = (β |C⟩ + α V̂ |B⟩) + α T̂ |B⟩
    mul!(C, A.V, B, α, β)
    A.to_p!(B)
    A.to_p!(C)
    mul!(C, A.T, B, α, true)
    A.to_x!(B)
    A.to_x!(C)
    #C̃ = similar(C)
    #mul!(C̃, A.T, B, α, false)
    #A.to_x!(C̃)
    #C .+= C̃
end


#### Split Generator


# Reminder: any generator needs to implement
#
# * getcontrols
# * evalcontrols
# * evalcontrols!
# * substitute_controls
# * getcontrolderiv
#

struct SplitGenerator
    T  # (potentially) time-dependent
    V  # time-dependent
    to_p!::Function
    to_x!::Function
end

function getcontrols(gen::SplitGenerator)
    @assert length(getcontrols(gen.T)) == 0
    return getcontrols(gen.V)
end

function evalcontrols(gen::SplitGenerator, args...)
    SplitOperator(
        evalcontrols(gen.T, args...),
        evalcontrols(gen.V, args...),
        gen.to_p!,
        gen.to_x!
    )
end

function evalcontrols!(op::SplitOperator, gen::SplitGenerator, args...)
    evalcontrols!(op.T, gen.T, args...)
    evalcontrols!(op.V, gen.V, args...)
end

function substitute_controls(gen::SplitGenerator, controls_map)
    @assert length(getcontrols(gen.T)) == 0
    V = substitute_controls(gen.V, controls_map)
    return SplitGenerator(gen.T, V, gen.to_p!, gen.to_x!)
end

function getcontrolderiv(generator::SplitGenerator, control)
    @assert length(getcontrols(gen.T)) == 0
    V_deriv = getcontrolderiv(generator.V, control)
    return SplitGenerator(gen.T, V_deriv, gen.to_p!, gen.to_x!)
end


#### RotTAI_PotentialGenerator


@doc raw"""
```math
V(t) = V_0 \cos(m(θ ± ϕ(t))
```
"""
@Base.kwdef struct RotTAI_PotentialGenerator
    V0::Float64
    m::Int64
    theta::Vector{Float64}
    phi # control
end


function getcontrols(gen::RotTAI_PotentialGenerator)
    return getcontrols(gen.phi)
end

function evalcontrols(gen::RotTAI_PotentialGenerator, vals_dict, args...)
    op = Diagonal(similar(gen.theta))
    evalcontrols!(op, gen, vals_dict, args...)
end


function evalcontrols!(
    op::Diagonal{Float64,Vector{Float64}},
    gen::RotTAI_PotentialGenerator,
    vals_dict,
    args...
)
    V₀::Float64 = gen.V0
    m::Int64 = gen.m
    θ::Vector{Float64} = gen.theta
    ϕ::Float64 = evalcontrols(gen.phi, vals_dict, args...)
    op.diag .= V₀ .* cos.(m .* (θ .- ϕ))
    return op
end


function getcontrolderiv(generator::RotTAI_PotentialGenerator, control)
    ∂ϕ = getcontrolderiv(generator.phi, control)
    if ∂ϕ == 0
        return nothing
    else
        return RotTAI_PotentialDerivGenerator(
            generator.V0,
            generator.m,
            generator.theta,
            generator.phi,
            ∂ϕ
        )
    end
end


#### RotTAI_PotentialDerivGenerator


@doc raw"""
```math
V(t) = - m V_0 \sin(m(θ ± ϕ(t))
```
"""
struct RotTAI_PotentialDerivGenerator
    V0::Float64
    m::Int64
    theta::Vector{Float64}
    phi # amplitude
    phi_deriv  # derivative of amplitude (may be 1.0)
end


function evalcontrols(gen::RotTAI_PotentialDerivGenerator, vals_dict, args...)
    op = Diagonal(similar(theta))
    evalcontrols!(op, gen, vals_dict, args...)
end


function evalcontrols!(
    op::Diagonal{Float64,Vector{Float64}},
    gen::RotTAI_PotentialDerivGenerator,
    vals_dict,
    args...
)
    V₀::Float64 = op.V0
    m::Int64 = op.m
    θ::Vector{Float64} = op.theta
    ϕ::Float64 = evalcontrols(op.phi, vals_dict, args...)
    ∂ϕ::Float64 = evalcontrols(op.phi_deriv, vals_dict, args...)
    op.diag .= -m .* V₀ .* ∂ϕ .* sin.(m .* (θ .+ ϕ))
end


#### get_ground_state


"""Determine a local ground state of the given operator Ĥ₀.

The `θ₀` and `d` should be approximate guesses for where the state should be
located and its width. That is, `θ₀` should be around the minimum of the well
for which the ground state should be obtained.

The state is obtained with imaginary split propagation with the given number of
`steps`.
"""
function get_ground_state(Ĥ₀::SplitOperator, theta_grid, θ₀=2π/16; steps=10000, d=0.05)
    h = -1im
    Uk2 = exp(-0.5im * h * Ĥ₀.T)
    Uk = exp(-1im * h * Ĥ₀.T)
    Ux = exp(-1im * h * Ĥ₀.V)
    θ = theta_grid

    Ψx = convert(Array{ComplexF64}, exp.(-(θ .- θ₀).^2/d^2))
    normalize!(Ψx)

    Ψk = fft(Ψx)

    for i=1:steps
        Ψk = Uk2 * Ψk
        Ψx = ifft(Ψk)
        Ψx = Ux * Ψx
        Ψk = fft(Ψx)
        Ψk = Uk2 * Ψk

        normalize!(Ψk)
    end

    Ψx = ifft(Ψk)
    normalize!(Ψx)

    return Ψx
end
