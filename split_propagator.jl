using QuantumPropagators:
    PWCPropagator, _pwc_set_t!, _pwc_set_genop!, _pwc_get_max_genop, _pwc_process_parameters, _pwc_advance_time!
import QuantumPropagators: init_prop, set_t!, prop_step!


###############################################################################

struct SplitPropWrk

    dt::Float64
    UT_op::Diagonal{ComplexF64, Vector{ComplexF64}}
    UV2_op::Diagonal{ComplexF64, Vector{ComplexF64}}
    T_is_static::Bool
    V_is_static::Bool

    function SplitPropWrk(gen::SplitGenerator, genop::SplitOperator, dt::Float64)
        UT_op = exp(-1im .* genop.T .* dt)
        UV2_op = exp(-0.5im .* genop.V .* dt)
        T_is_static = (gen.T ≡ genop.T)
        V_is_static = (gen.V ≡ genop.V)
        new(dt, UT_op, UV2_op, T_is_static, V_is_static)
    end
end


function splitprop!(Ψ, H::SplitOperator, dt, wrk; _...)
    #=@assert dt ≈ wrk.dt "dt=$dt ≠ wrk.dt=$(wrk.dt)"=#
    T = H.T
    V = H.V
    if !wrk.T_is_static
        wrk.UT_op.diag .= exp.(-1im .* T.diag .* dt)
    end
    if !wrk.V_is_static
        wrk.UV2_op.diag .= exp.(-0.5im .* V.diag .* dt)
    end
    @. Ψ = wrk.UV2_op.diag * Ψ
    H.to_p!(Ψ)
    @. Ψ = wrk.UT_op.diag * Ψ
    H.to_x!(Ψ)
    @. Ψ = wrk.UV2_op.diag * Ψ
    return Ψ
end


###############################################################################

mutable struct SplitPropagator{GT,OT,ST,WT<:SplitPropWrk} <: PWCPropagator
    generator::GT
    state::ST
    t::Float64  # time at which current `state` is defined
    n::Int64 # index of next interval to propagate
    tlist::Vector{Float64}
    parameters::AbstractDict
    controls
    genop::OT
    wrk::WT
    backward::Bool
    inplace::Bool
end


set_t!(propagator::SplitPropagator, t) = _pwc_set_t!(propagator, t)

function init_prop(
    state,
    generator,
    tlist,
    method::Val{:splitprop};
    inplace=true,
    backward=false,
    verbose=false,
    parameters=nothing,
    _...
)
    generator::SplitGenerator
    tlist = convert(Vector{Float64}, tlist)
    controls = get_controls(generator)
    G::SplitOperator = _pwc_get_max_genop(generator, controls, tlist)
    parameters = _pwc_process_parameters(parameters, controls, tlist)
    n = 1
    t = tlist[1]
    if backward
        n = length(tlist) - 1
        t = float(tlist[n+1])
    end
    dt = tlist[2] - tlist[1]
    if backward
        dt = -dt
    end
    wrk = SplitPropWrk(generator, G, dt)
    GT = typeof(generator)
    OT = typeof(G)
    ST = typeof(state)
    WT = typeof(wrk)
    return SplitPropagator{GT,OT,ST,WT}(
        generator,
        inplace ? copy(state) : state,
        t,
        n,
        tlist,
        parameters,
        controls,
        G,
        wrk,
        backward,
        inplace,
    )
end


function prop_step!(propagator::SplitPropagator)
    n = propagator.n
    tlist = getfield(propagator, :tlist)
    (0 < n < length(tlist)) || return nothing
    dt = tlist[n+1] - tlist[n]
    if propagator.backward
        dt = -dt
    end
    Ψ = propagator.state
    if propagator.inplace
        _pwc_set_genop!(propagator, n)
        H = propagator.genop
        _Ψ = splitprop!(Ψ, H, dt, propagator.wrk)
        @assert _Ψ ≡ Ψ # DEBUG
    else
        error("Not implemented")
    end
    _pwc_advance_time!(propagator)
    return propagator.state
end
