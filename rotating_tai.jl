using LinearAlgebra
using QuantumPropagators
using QuantumPropagators: Operator, Generator
import QuantumPropagators.Controls:
    get_controls, evaluate, evaluate!, substitute
import QuantumControlBase: get_control_deriv, dynamical_generator_adjoint


#### Split Operator


# Reminder: any operator needs to implement
#
# * mul!

struct SplitOperator{TT,TV}
    T::TT
    V::TV
    to_p!::Function # coord to momentum
    to_x!::Function # momentum to coord
    function SplitOperator(T, V, to_p!, to_x!)
        T::Union{Nothing, Diagonal{Float64,Vector{Float64}}}
        V::Union{Nothing, Diagonal{Float64,Vector{Float64}}}
        # ishermitian depends on these type-asserts
        new{typeof(T),typeof(V)}(T, V, to_p!, to_x!)
    end
end


Base.size(O::SplitOperator{TT,VT}) where {TT,VT} = size(O.T)
Base.size(O::SplitOperator{Nothing, VT}) where {VT} = size(O.V)
Base.size(O::SplitOperator{Nothing,Nothing}) = 0


LinearAlgebra.ishermitian(o::SplitOperator) = true


function LinearAlgebra.mul!(C, A::SplitOperator, B, α, β)
    # |C⟩ = β |C⟩ + α Â |B⟩ = (β |C⟩ + α V̂ |B⟩) + α T̂ |B⟩
    mul!(C, A.V, B, α, β)
    A.to_p!(B)
    A.to_p!(C)
    mul!(C, A.T, B, α, true)
    A.to_x!(B)
    A.to_x!(C)
    return C
end

# Potential only
function LinearAlgebra.mul!(C, A::SplitOperator{Nothing, TV}, B, α, β) where {TV}
    mul!(C, A.V, B, α, β)
end

# Momentum space operator only
function LinearAlgebra.mul!(C, A::SplitOperator{TT, Nothing}, B, α, β) where {TT}
    A.to_p!(B)
    A.to_p!(C)
    mul!(C, A.T, B, α, β)
    A.to_x!(B)
    A.to_x!(C)
    return C
end

# Zero operator
function LinearAlgebra.mul!(C, A::SplitOperator{Nothing, Nothing}, B, α, β)
    lmul!(β, C)
end


function Base.:*(H::SplitOperator, Ψ)
    # TODO: it would be better of have a specialized dot
    ϕ = similar(Ψ)
    LinearAlgebra.mul!(ϕ, H, Ψ, true, false)
    return ϕ
end


get_controls(::SplitOperator) = ( );

evaluate(O::SplitOperator, args...; kwargs...) = O;

evaluate!(O::SplitOperator, H::SplitOperator, args...; kwargs...) = O;


#### Split Generator


# Reminder: any generator needs to implement
#
# * get_controls
# * evaluate
# * evaluate!
# * substitute
# * get_control_deriv
#

struct SplitGenerator
    T  # (potentially) time-dependent
    V  # time-dependent
    to_p!::Function
    to_x!::Function
end

function get_controls(gen::SplitGenerator)
    if !isnothing(gen.T) && !isnothing(gen.V)
        return (get_controls(gen.T)..., get_controls(gen.V)...)
    elseif isnothing(gen.T) && !isnothing(gen.V)
        return get_controls(gen.V)
    elseif !isnothing(gen.T) && isnothing(gen.V)
        return get_controls(gen.T)
    else
        return ()
    end
end

function evaluate(gen::SplitGenerator, args...; kwargs...)
    T̂ = isnothing(gen.T) ? nothing : evaluate(gen.T, args...; kwargs...)
    if T̂ isa Operator
        # SplitOperator can only have a "Diagonal" matrix. If we get a general
        # operator, we have to sum it into a single operator
        if (length(T̂.ops) == 2) && (length(T̂.coeffs) == 1)
            T̂ = T̂.ops[1] + T̂.coeffs[1] * T̂.ops[2]
        elseif (length(T̂.ops) == 1) && (length(T̂.coeffs) == 1)
            T̂ = T̂.coeffs[1] * T̂.ops[1]
        else
            error("Not implemented")
        end
    end
    V̂ = isnothing(gen.V) ? nothing : evaluate(gen.V, args...; kwargs...)
    SplitOperator(T̂, V̂, gen.to_p!, gen.to_x!)
end


function evaluate!(
    op::Diagonal{Float64, Vector{Float64}},
    gen::Generator{Diagonal{Float64, Vector{Float64}}, CT},
    tlist::Vector{Float64},
    n::Int64;
    vals_dict=IdDict(),
) where {CT}
    if (length(gen.ops) == 2) && (length(gen.amplitudes) == 1)
        op.diag .= gen.ops[1].diag
        val = evaluate(gen.amplitudes[1], tlist, n; vals_dict)
        op.diag .= op.diag .+ val .* gen.ops[2].diag
    elseif (length(gen.ops) == 1) && (length(gen.amplitudes) == 1)
        val = evaluate(gen.amplitudes[1], tlist, n; vals_dict)
        op.diag .= val .* gen.ops[1].diag
    else
        error("Not implemented")
    end
end


function evaluate!(op::SplitOperator, gen::SplitGenerator, args...; kwargs...)
    if !isnothing(op.T)
        evaluate!(op.T, gen.T, args...; kwargs...)
    end
    if !isnothing(op.V)
        evaluate!(op.V, gen.V, args...; kwargs...)
    end
end


function substitute(gen::SplitGenerator, controls_map)
    @assert length(get_controls(gen.T)) == 0
    V = isnothing(gen.V) ? nothing : substitute(gen.V, controls_map)
    T = isnothing(gen.T) ? nothing : substitute(gen.T, controls_map)
    return SplitGenerator(T, V, gen.to_p!, gen.to_x!)
end


function get_control_deriv(gen::SplitGenerator, control)
    T_deriv = isnothing(gen.T) ? nothing : get_control_deriv(gen.T, control)
    V_deriv = isnothing(gen.V) ? nothing : get_control_deriv(gen.V, control)
    if isnothing(T_deriv) && isnothing(V_deriv)
        return nothing
    else
        if isnothing(T_deriv)
            return V_deriv
        else
            return SplitGenerator(T_deriv, V_deriv, gen.to_p!, gen.to_x!)
        end
    end
end

dynamical_generator_adjoint(G::SplitGenerator) = G


#### RotTAI_PotentialGenerator
#
# This gets plugged in to `SplitGenerator` as the V component


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


function get_controls(gen::RotTAI_PotentialGenerator)
    return get_controls(gen.phi)
end

function evaluate(gen::RotTAI_PotentialGenerator, args...; kwargs...)
    op = Diagonal(similar(gen.theta))
    evaluate!(op, gen, args...; kwargs...)
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


function evaluate!(
    op::Diagonal{Float64,Vector{Float64}},
    gen::RotTAI_PotentialGenerator,
    args...;
    kwargs...

)
    V₀::Float64 = gen.V0
    m::Int64 = gen.m
    θ::Vector{Float64} = gen.theta
    ϕ::Float64 = evaluate(gen.phi, args...; kwargs...)
    Ω::Float64 = gen.Omega
    Ω_t = 0.0
    if Ω ≠ 0.0
        Ω_t = evaluate(t -> Ω * t, args...; kwargs...)
    end
    if gen.direction > 0
        op.diag .= V₀ .* cos.(m .* (θ .- ϕ .+ Ω_t))
    else
        op.diag .= V₀ .* cos.(m .* (θ .+ ϕ .+ Ω_t))
    end
    return op
end


function get_control_deriv(generator::RotTAI_PotentialGenerator, control)
    ∂ϕ = get_control_deriv(generator.phi, control)
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


function evaluate(gen::RotTAI_PotentialDerivGenerator, args...; kwargs...)
    op = Diagonal(similar(gen.theta))
    evaluate!(op, gen, vals_dict, args...; kwargs...)
end


function evaluate!(
    op::Diagonal{Float64,Vector{Float64}},
    gen::RotTAI_PotentialDerivGenerator,
    args...;
    vals_dict=IdDict()
)
    V₀::Float64 = gen.V0
    m::Int64 = gen.m
    θ::Vector{Float64} = gen.theta
    Ω::Float64 = gen.Omega
    Ω_t = 0.0
    if Ω ≠ 0.0
        Ω_t = Ω * _t(tlist, n)
    end
    ϕ::Float64 = evaluate(gen.phi, args...; vals_dict)
    ∂ϕ::Float64 = evaluate(gen.phi_deriv, args...; vals_dict)
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

    Ψ = convert(Array{ComplexF64}, exp.(-(θ .- θ₀).^2/d^2))
    normalize!(Ψ)

    fft = plan_fft!(Ψ)
    ifft = plan_ifft!(Ψ)

    Ψ = fft * Ψ

    for i=1:steps
        @. Ψ = Uk2.diag * Ψ
        Ψ = ifft * Ψ
        @. Ψ = Ux.diag * Ψ
        Ψ = fft * Ψ
        @. Ψ = Uk2.diag * Ψ
        normalize!(Ψ)
    end

    Ψ = ifft * Ψ
    normalize!(Ψ)

    return Ψ

end
