using QuantumPropagators.Amplitudes: ControlAmplitude, LockedAmplitude
using QuantumPropagators.Controls: discretize_on_midpoints
import QuantumPropagators.Controls: evalcontrols
import QuantumControlBase: getcontrolderiv

"""An amplitude of the form ``G(t) + S(t) ϵ(t)``.

```julia
ampl = GuidedAmplitude(control; shape=shape, guide=guide)
ampl = GuidedAmplitude(control, tlist; shape=shape, guide=guide)
```
"""
abstract type GuidedAmplitude <: ControlAmplitude end

function GuidedAmplitude(control; shape, guide)
    if (control isa Vector{Float64}) && (shape isa Vector{Float64}) && (guide isa Vector{Float64})
        return GuidedPulseAmplitude(control, shape, guide)
    else
        try
            ϵ_t = control(0.0)
        catch
            error(
                "A GuidedAmplitude control must either be a vector of values or a callable"
            )
        end
        try
            S_t = shape(0.0)
        catch
            error("A GuidedAmplitude shape must either be a vector of values or a callable")
        end
        try
            G_t = guide(0.0)
        catch
            error("A GuidedAmplitude guide must either be a vector of values or a callable")
        end
        return GuidedContinuousAmplitude(control, shape, guide)
    end
end


function GuidedAmplitude(control, tlist; shape, guide)
    control = discretize_on_midpoints(control, tlist)
    shape = discretize_on_midpoints(shape, tlist)
    guide = discretize_on_midpoints(guide, tlist)
    return GuidedPulseAmplitude(control, shape, guide)
end


function Base.show(io::IO, ampl::GuidedAmplitude)
    print(io, "GuidedAmplitude(::$(typeof(ampl.control)); guide::$(typeof(ampl.guide)), shape::$(typeof(ampl.shape)))")
end


struct GuidedPulseAmplitude <: GuidedAmplitude
    control::Vector{Float64}
    shape::Vector{Float64}
    guide::Vector{Float64}
end


function Base.Array(ampl::GuidedPulseAmplitude)
    ampl.guide .+ ampl.shape .* ampl.control
end


struct GuidedContinuousAmplitude <: GuidedAmplitude
    control
    shape
    guide
end

(ampl::GuidedContinuousAmplitude)(t::Float64) = ampl.guide(t) + ampl.shape(t) * ampl.control(t)


function evalcontrols(ampl::GuidedPulseAmplitude, vals_dict, tlist, n)
    return ampl.shape[n] * vals_dict[ampl.control] + ampl.guide[n]
end


function evalcontrols(ampl::GuidedContinuousAmplitude, vals_dict, tlist, n)
    error("Not implemented")
end

getcontrolderiv(ampl::GuidedAmplitude, control) =
    (control ≡ ampl.control) ? LockedAmplitude(ampl.shape) : 0.0

