using LinearAlgebra
using QuantumPropagators
using QuantumPropagators: Operator, Generator
import QuantumPropagators.Controls:
    getcontrols, evalcontrols, evalcontrols!, substitute_controls
import QuantumControlBase: getcontrolderiv, dynamical_generator_adjoint


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
    return C
end


function Base.:*(H::SplitOperator, Ψ)
    # TODO: it would be better of have a specialized dot
    ϕ = similar(Ψ)
    LinearAlgebra.mul!(ϕ, H, Ψ, true, true)
    return ϕ
end


getcontrols(::SplitOperator) = ( );

evalcontrols(O::SplitOperator, args...) = O;

evalcontrols!(O::SplitOperator, H::SplitOperator, args...) = O;


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
    return (getcontrols(gen.T)..., getcontrols(gen.V)...)
end

function evalcontrols(gen::SplitGenerator, args...)
    T̂ = evalcontrols(gen.T, args...)
    if T̂ isa Operator
        @assert (length(T̂.ops) == 2) && (length(T̂.coeffs) == 1)
        T̂ = T̂.ops[1] + T̂.coeffs[1] * T̂.ops[2]
    end
    V̂ = evalcontrols(gen.V, args...)
    SplitOperator(T̂, V̂, gen.to_p!, gen.to_x!)
end


function evalcontrols!(
    op::Diagonal{Float64, Vector{Float64}},
    gen::Generator{Diagonal{Float64, AbstractVector{Float64}}, Vector{Float64}},
    vals_dict,
    tlist::Vector{Float64},
    n::Int64
)
    @assert (length(gen.ops) == 2) && (length(gen.amplitudes) == 1)
    op.diag .= gen.ops[1].diag
    val = vals_dict[gen.amplitudes[1]]
    op.diag .= op.diag .+ val .* gen.ops[2].diag
end

function evalcontrols!(
    op::Diagonal{Float64, Vector{Float64}},
    gen::Generator{Diagonal{Float64, Vector{Float64}}, Vector{Float64}},
    vals_dict,
    tlist::Vector{Float64},
    n::Int64
)
    @assert (length(gen.ops) == 2) && (length(gen.amplitudes) == 1)
    op.diag .= gen.ops[1].diag
    val = vals_dict[gen.amplitudes[1]]
    op.diag .= op.diag .+ val .* gen.ops[2].diag
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

function getcontrolderiv(gen::SplitGenerator, control)
    @assert length(getcontrols(gen.T)) == 0
    V_deriv = getcontrolderiv(gen.V, control)
    return SplitGenerator(gen.T, V_deriv, gen.to_p!, gen.to_x!)
end

dynamical_generator_adjoint(G::SplitGenerator) = G


#### RotTAI_PotentialGenerator


@doc raw"""
```math
V(t) = V_0 \cos(m(θ ± ϕ(t) + Ωt)
```
"""
struct RotTAI_PotentialGenerator
    V0::Float64
    m::Int64
    theta::Vector{Float64}
    phi # control
    Omega::Float64
    direction::Int64
    function RotTAI_PotentialGenerator(;V0, m, theta, phi, Omega=0.0, direction=1)
        new(V0, m, theta, phi, Omega, direction)
    end
end


function getcontrols(gen::RotTAI_PotentialGenerator)
    return getcontrols(gen.phi)
end

function evalcontrols(gen::RotTAI_PotentialGenerator, vals_dict, args...)
    op = Diagonal(similar(gen.theta))
    evalcontrols!(op, gen, vals_dict, args...)
end


# Midpoint of n'th interval of tlist, but snap to beginning/end (that's
# because any S(t) is likely exactly zero at the beginning and end, and we
# want to use that value for the first and last time interval)
function _t(tlist, n)
    @assert 1 <= n <= (length(tlist) - 1)  # n is an *interval* of `tlist`
    if n == 1
        t = tlist[begin]
    elseif n == length(tlist) - 1
        t = tlist[end]
    else
        dt = tlist[n+1] - tlist[n]
        t = tlist[n] + dt / 2
    end
    return t
end


function evalcontrols!(
    op::Diagonal{Float64,Vector{Float64}},
    gen::RotTAI_PotentialGenerator,
    vals_dict,
    tlist,
    n
)
    V₀::Float64 = gen.V0
    m::Int64 = gen.m
    θ::Vector{Float64} = gen.theta
    ϕ::Float64 = evalcontrols(gen.phi, vals_dict, tlist, n)
    Ω::Float64 = gen.Omega
    Ω_t = 0.0
    if Ω ≠ 0.0
        Ω_t = Ω * _t(tlist, n)
    end
    if gen.direction > 0
        op.diag .= V₀ .* cos.(m .* (θ .- ϕ .+ Ω_t))
    else
        op.diag .= V₀ .* cos.(m .* (θ .+ ϕ .+ Ω_t))
    end
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
            ∂ϕ,
            generator.Omega,
            generator.direction
        )
    end
end


#### RotTAI_PotentialDerivGenerator


struct RotTAI_PotentialDerivGenerator
    V0::Float64
    m::Int64
    theta::Vector{Float64}
    phi # amplitude
    phi_deriv  # derivative of amplitude (should be 1.0)
    Omega::Float64
    direction::Int64
end


function evalcontrols(gen::RotTAI_PotentialDerivGenerator, vals_dict, args...)
    op = Diagonal(similar(gen.theta))
    evalcontrols!(op, gen, vals_dict, args...)
end


function evalcontrols!(
    op::Diagonal{Float64,Vector{Float64}},
    gen::RotTAI_PotentialDerivGenerator,
    vals_dict,
    args...
)
    V₀::Float64 = gen.V0
    m::Int64 = gen.m
    θ::Vector{Float64} = gen.theta
    Ω::Float64 = gen.Omega
    Ω_t = 0.0
    if Ω ≠ 0.0
        Ω_t = Ω * _t(tlist, n)
    end
    ϕ::Float64 = evalcontrols(gen.phi, vals_dict, args...)
    ∂ϕ::Float64 = evalcontrols(gen.phi_deriv, vals_dict, args...)
    @assert ∂ϕ == gen.phi_deriv == 1.0
    if gen.direction > 0
        op.diag .= -m .* V₀ .* ∂ϕ .* sin.(m .* (θ .- ϕ .+ Ω_t))
    else
        op.diag .= -m .* V₀ .* ∂ϕ .* sin.(m .* (θ .+ ϕ .+ Ω_t))
    end
    return op
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
