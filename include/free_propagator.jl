using LinearAlgebra
using QuantumPropagators: AbstractPropagator
import QuantumPropagators: init_prop, set_t!, prop_step!

###############################################################################

struct FreePropWrk{ST,UT,ET}
    ϕ::ST
    U::UT
    U_inv::UT
    λ::ET
    function FreePropWrk(Ψ::ST, H) where {ST}
        λ, U = eigen(H)
        UT = typeof(U)
        ET = typeof(λ)
        if ishermitian(H)
            U_inv = copy(U')
        else
            U_inv = inv(U)
        end
        new{ST,UT,ET}(similar(Ψ), U, U_inv, λ)
    end
end

function freeprop!(Ψ, dt, wrk)
    mul!(wrk.ϕ, wrk.U_inv, Ψ)
    wrk.ϕ .= exp.(-1im .* wrk.λ .* dt) .* wrk.ϕ
    mul!(Ψ, wrk.U, wrk.ϕ)
end

function freeprop(Ψ, dt, wrk)
    return wrk.U * (exp.(-1im .* wrk.λ .* dt) .* (wrk.U_inv * Ψ))
end

###############################################################################

mutable struct FreePropagator{GT,ST,WT} <: AbstractPropagator
    generator::GT
    state::ST
    t::Float64  # time at which current `state` is defined
    n::Int64 # index of next interval to propagate
    tlist::Vector{Float64}
    parameters::Nothing
    wrk::WT
    backward::Bool
    inplace::Bool
end


set_t!(propagator::FreePropagator, t) = setfield!(propagator, :t, t)

function init_prop(
    state,
    generator,
    tlist,
    method::Val{:freeprop};
    inplace=true,
    backward=false,
    verbose=false,
    parameters=nothing,
    _...
)
    generator::AbstractMatrix
    tlist = convert(Vector{Float64}, tlist)
    wrk =  FreePropWrk(state, generator)
    GT = typeof(generator)
    ST = typeof(state)
    WT = typeof(wrk)
    n = 1
    t = tlist[1]
    if backward
        n = length(tlist) - 1
        t = float(tlist[n+1])
    end
    return FreePropagator{GT,ST,WT}(
        generator,
        inplace ? copy(state) : state,
        t,
        n,
        tlist,
        parameters,
        wrk,
        backward,
        inplace,
    )
end


function prop_step!(propagator::FreePropagator)
    n = propagator.n
    tlist = getfield(propagator, :tlist)
    (0 < n < length(tlist)) || return nothing
    dt = tlist[n+1] - tlist[n]
    if propagator.backward
        dt = -dt
    end
    Ψ = propagator.state
    if propagator.inplace
        _Ψ = freeprop!(Ψ, dt, propagator.wrk)
        @assert _Ψ ≡ Ψ # DEBUG
    else
        _Ψ = freeprop(Ψ, dt, propagator.wrk)
        setfield!(propagator, :state, _Ψ)
        @assert _Ψ ≢ Ψ # DEBUG
    end
    if propagator.backward
        setfield!(propagator, :t, tlist[n])
        setfield!(propagator, :n, n - 1)
    else
        setfield!(propagator, :t, tlist[n+1])
        setfield!(propagator, :n, n + 1)
    end
    return propagator.state
end
