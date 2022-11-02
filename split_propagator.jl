using QuantumPropagators: PWCPropagator, _pwc_set_t!, _pwc_set_genop!, _pwc_get_max_genop, _pwc_process_parameters

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

function initprop(
    state,
    generator::SplitGenerator,
    tlist,
    method::Val{:splitprop};
    inplace=true,
    backward=false,
    verbose=false,
    parameters=nothing,
    _...
)
    tlist = convert(Vector{Float64}, tlist)
    controls = getcontrols(generator)
    G::SplitOperator = _pwc_get_max_genop(generator, controls, tlist)
    parameters = _pwc_process_parameters(parameters, controls, tlist)
    wrk = SplitPropWrk(...)
    n = 1
    t = tlist[1]
    if backward
        n = length(tlist) - 1
        t = float(tlist[n+1])
    end
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


function propstep!(propagator::SplitPropagator)
    n = propagator.n
    tlist = getfield(propagator, :tlist)
    (0 < n < length(tlist)) || return nothing
    dt = tlist[n+1] - tlist[n]
    if propagator.backward
        dt = -dt
    end
    if propagator.inplace
    else
        _pwc_set_genop!(propagator, n)
        error("Not implemented")
    end
end


###############################################################################

struct SplitPropWrk
    function SplitPropWrk()
        new()
    end
end


function splitprop!(Ψ, H::SplitOperator, dt, wrk; _...)
    # TODO
end
